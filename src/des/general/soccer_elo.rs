//! Relative policy evaluation: Elo ratings + a cross-play payoff matrix.
//!
//! Absolute fitness (goals scored, win count vs a fixed opponent) is a noisy,
//! non-stationary signal here: matches are per-process nondeterministic (the
//! formation LP — see the determinism audit) and the opponent field drifts as
//! self-play advances, so the same brain can post different "fitness" run to
//! run. A policy's strength is better expressed *relative to the field*:
//!
//!   * [`EloRatings`] — an online logistic rating folded from head-to-head
//!     results. Robust to which specific opponents a brain happened to draw.
//!   * [`CrossPlayMatrix`] — the order-independent empirical payoff `P[i][j]`
//!     (mean score of `i` against `j`). This is the substrate a population /
//!     PSRO meta-solver consumes, and it exposes *exploitability*: a brain that
//!     beats the current champion but loses to an older one is not actually
//!     better, only different.
//!
//! Pure + deterministic given its inputs; no engine or RNG dependency. Folds
//! directly from [`MatchReport`]s produced by the tournament runner.

use std::collections::BTreeMap;

use crate::des::general::tournament::MatchReport;

/// Conventional Elo anchor for an unseen competitor.
pub const ELO_DEFAULT_RATING: f64 = 1500.0;
/// Default K-factor (update step). Moderate: fast enough to track a learner,
/// damped enough not to swing on a single noisy result.
pub const ELO_DEFAULT_K: f64 = 24.0;

/// Expected score for A against B under the logistic Elo model: A's win
/// probability plus half its draw probability, in `(0, 1)`. Symmetric:
/// `expected(a, b) + expected(b, a) == 1`.
pub fn elo_expected_score(rating_a: f64, rating_b: f64) -> f64 {
    let ra = if rating_a.is_finite() {
        rating_a
    } else {
        ELO_DEFAULT_RATING
    };
    let rb = if rating_b.is_finite() {
        rating_b
    } else {
        ELO_DEFAULT_RATING
    };
    1.0 / (1.0 + 10.0_f64.powf((rb - ra) / 400.0))
}

/// Score of a match from the home/first side's perspective: `1.0` win, `0.5`
/// draw, `0.0` loss. A knockout shootout resolves a level score to the side the
/// shootout awarded (so a drawn-then-won tie counts as a win for the winner).
pub fn match_score(home_goals: u32, away_goals: u32, home_won_shootout: Option<bool>) -> f64 {
    match home_goals.cmp(&away_goals) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => 0.0,
        std::cmp::Ordering::Equal => match home_won_shootout {
            Some(true) => 1.0,
            Some(false) => 0.0,
            None => 0.5,
        },
    }
}

/// Online Elo ratings keyed by team/policy id. `BTreeMap` so iteration order
/// (and thus the derived ranking on ties) is deterministic.
#[derive(Clone, Debug)]
pub struct EloRatings {
    ratings: BTreeMap<usize, f64>,
    k: f64,
}

impl Default for EloRatings {
    fn default() -> Self {
        Self::new(ELO_DEFAULT_K)
    }
}

impl EloRatings {
    pub fn new(k: f64) -> Self {
        let k = if k.is_finite() && k > 0.0 {
            k
        } else {
            ELO_DEFAULT_K
        };
        EloRatings {
            ratings: BTreeMap::new(),
            k,
        }
    }

    /// Current rating for `id`, or the default anchor if it has never played.
    pub fn rating(&self, id: usize) -> f64 {
        self.ratings.get(&id).copied().unwrap_or(ELO_DEFAULT_RATING)
    }

    /// Apply one head-to-head result: `score_a` is A's outcome in `[0, 1]`
    /// (see [`match_score`]). Zero-sum — B's rating moves by the opposite delta,
    /// so the population's mean rating is conserved.
    pub fn update(&mut self, id_a: usize, id_b: usize, score_a: f64) {
        if id_a == id_b {
            return;
        }
        let score_a = score_a.clamp(0.0, 1.0);
        let ra = self.rating(id_a);
        let rb = self.rating(id_b);
        let expected_a = elo_expected_score(ra, rb);
        let delta = self.k * (score_a - expected_a);
        self.ratings.insert(id_a, ra + delta);
        self.ratings.insert(id_b, rb - delta);
    }

    /// Fold a batch of fixtures, sweeping the list `passes` times. A single pass
    /// is order-dependent (early games update against stale ratings); repeated
    /// sweeps relax the estimate toward the maximum-likelihood ratings the games
    /// imply. `passes` is clamped to at least 1.
    pub fn from_match_reports(reports: &[MatchReport], k: f64, passes: usize) -> Self {
        let mut ratings = EloRatings::new(k);
        let passes = passes.max(1);
        for _ in 0..passes {
            for report in reports {
                let home_won_shootout = report
                    .shootout_winner
                    .map(|winner| winner == report.home_id);
                let score = match_score(report.home_goals, report.away_goals, home_won_shootout);
                ratings.update(report.home_id, report.away_id, score);
            }
        }
        ratings
    }

    /// Ids ranked best-first by rating (ties broken by ascending id for
    /// determinism).
    pub fn ranked(&self) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = self.ratings.iter().map(|(&id, &r)| (id, r)).collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked
    }

    /// The top-rated id, or `None` if no games have been folded.
    pub fn leader(&self) -> Option<usize> {
        self.ranked().first().map(|(id, _)| *id)
    }
}

/// Running win/draw/game tally for one ordered pair `(i, j)`.
#[derive(Clone, Copy, Debug, Default)]
struct PairTally {
    /// Total score `i` accumulated against `j` (win = 1, draw = 0.5).
    score: f64,
    games: u32,
}

/// Empirical cross-play payoffs: for every ordered pair of ids that has met,
/// the mean score of the first against the second. Order-independent (it is an
/// average, not a sequential update), so unlike raw Elo it does not depend on
/// fixture ordering — the artifact a population meta-solver and an
/// exploitability check consume.
#[derive(Clone, Debug, Default)]
pub struct CrossPlayMatrix {
    /// `pairs[(i, j)]` = `i`'s record against `j`. Both `(i, j)` and `(j, i)`
    /// are recorded so lookups need no canonical ordering.
    pairs: BTreeMap<(usize, usize), PairTally>,
}

impl CrossPlayMatrix {
    pub fn new() -> Self {
        CrossPlayMatrix::default()
    }

    /// Record one fixture (both orientations).
    pub fn record(&mut self, id_a: usize, id_b: usize, score_a: f64) {
        if id_a == id_b {
            return;
        }
        let score_a = score_a.clamp(0.0, 1.0);
        let a = self.pairs.entry((id_a, id_b)).or_default();
        a.score += score_a;
        a.games += 1;
        let b = self.pairs.entry((id_b, id_a)).or_default();
        b.score += 1.0 - score_a;
        b.games += 1;
    }

    pub fn from_match_reports(reports: &[MatchReport]) -> Self {
        let mut matrix = CrossPlayMatrix::new();
        for report in reports {
            let home_won_shootout = report
                .shootout_winner
                .map(|winner| winner == report.home_id);
            let score = match_score(report.home_goals, report.away_goals, home_won_shootout);
            matrix.record(report.home_id, report.away_id, score);
        }
        matrix
    }

    /// Mean score of `id_a` against `id_b`, or `None` if they have never met.
    /// `0.5` = even, `> 0.5` = `a` has the edge.
    pub fn payoff(&self, id_a: usize, id_b: usize) -> Option<f64> {
        self.pairs.get(&(id_a, id_b)).and_then(|t| {
            if t.games == 0 {
                None
            } else {
                Some(t.score / t.games as f64)
            }
        })
    }

    /// Games played between `id_a` and `id_b` (either orientation collapses to
    /// the same count).
    pub fn games_between(&self, id_a: usize, id_b: usize) -> u32 {
        self.pairs.get(&(id_a, id_b)).map(|t| t.games).unwrap_or(0)
    }

    /// All distinct competitor ids that appear in the matrix (sorted).
    pub fn ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.pairs.keys().flat_map(|&(a, b)| [a, b]).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Mean payoff of `id` against the rest of the field — its uniform-meta
    /// strength. `None` if `id` has played no one. This is the noise-robust
    /// "how good is this brain overall" scalar that replaces absolute fitness.
    pub fn mean_payoff_vs_field(&self, id: usize) -> Option<f64> {
        let mut total = 0.0;
        let mut opponents = 0u32;
        for other in self.ids() {
            if other == id {
                continue;
            }
            if let Some(p) = self.payoff(id, other) {
                total += p;
                opponents += 1;
            }
        }
        if opponents == 0 {
            None
        } else {
            Some(total / opponents as f64)
        }
    }

    /// Worst-case payoff of `id` against any single opponent it has faced — an
    /// *exploitability* read. A brain with a high field mean but a low worst
    /// case (some opponent reliably beats it) is fragile: a counter exists. The
    /// returned `(opponent, payoff)` is the most exploitative matchup.
    pub fn worst_case(&self, id: usize) -> Option<(usize, f64)> {
        self.ids()
            .into_iter()
            .filter(|&other| other != id)
            .filter_map(|other| self.payoff(id, other).map(|p| (other, p)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::tournament::TournamentStage;

    fn report(home_id: usize, away_id: usize, hg: u32, ag: u32) -> MatchReport {
        MatchReport {
            stage: TournamentStage::Group,
            home_id,
            away_id,
            home_name: format!("t{home_id}"),
            away_name: format!("t{away_id}"),
            home_goals: hg,
            away_goals: ag,
            shootout_winner: None,
            home_training_steps: 0,
            away_training_steps: 0,
        }
    }

    #[test]
    fn expected_score_is_symmetric_and_monotone() {
        assert!((elo_expected_score(1500.0, 1500.0) - 0.5).abs() < 1e-9);
        let stronger = elo_expected_score(1700.0, 1500.0);
        assert!(stronger > 0.5);
        // Symmetry: the two perspectives sum to one.
        assert!((stronger + elo_expected_score(1500.0, 1700.0) - 1.0).abs() < 1e-9);
        // Non-finite inputs fall back to the anchor rather than producing NaN.
        assert!((elo_expected_score(f64::NAN, 1500.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn match_score_handles_win_draw_loss_and_shootout() {
        assert_eq!(match_score(2, 1, None), 1.0);
        assert_eq!(match_score(0, 3, None), 0.0);
        assert_eq!(match_score(1, 1, None), 0.5);
        assert_eq!(match_score(1, 1, Some(true)), 1.0);
        assert_eq!(match_score(1, 1, Some(false)), 0.0);
    }

    #[test]
    fn a_win_raises_the_winner_and_lowers_the_loser_zero_sum() {
        let mut elo = EloRatings::new(ELO_DEFAULT_K);
        elo.update(1, 2, 1.0);
        assert!(elo.rating(1) > ELO_DEFAULT_RATING);
        assert!(elo.rating(2) < ELO_DEFAULT_RATING);
        // Zero-sum: total rating conserved.
        assert!((elo.rating(1) + elo.rating(2) - 2.0 * ELO_DEFAULT_RATING).abs() < 1e-9);
    }

    #[test]
    fn consistent_winner_climbs_above_the_field() {
        // Team 1 beats everyone; team 4 loses to everyone.
        let reports = vec![
            report(1, 2, 2, 0),
            report(1, 3, 1, 0),
            report(1, 4, 3, 0),
            report(2, 3, 1, 0),
            report(2, 4, 2, 0),
            report(3, 4, 1, 0),
        ];
        let elo = EloRatings::from_match_reports(&reports, ELO_DEFAULT_K, 12);
        let ranked = elo.ranked();
        assert_eq!(
            ranked.first().unwrap().0,
            1,
            "the team that beat everyone leads"
        );
        assert_eq!(
            ranked.last().unwrap().0,
            4,
            "the team that lost to everyone trails"
        );
        assert_eq!(elo.leader(), Some(1));
    }

    #[test]
    fn cross_play_payoff_is_order_independent_and_averages() {
        // Same fixtures in two different orders must yield identical payoffs
        // (unlike sequential Elo, this is an average).
        let a = vec![report(1, 2, 1, 0), report(1, 2, 0, 1)];
        let b = vec![report(1, 2, 0, 1), report(1, 2, 1, 0)];
        let ma = CrossPlayMatrix::from_match_reports(&a);
        let mb = CrossPlayMatrix::from_match_reports(&b);
        // One win, one loss apiece ⇒ even payoff both ways.
        assert_eq!(ma.payoff(1, 2), Some(0.5));
        assert_eq!(ma.payoff(1, 2), mb.payoff(1, 2));
        assert_eq!(ma.payoff(2, 1), Some(0.5));
        assert_eq!(ma.games_between(1, 2), 2);
    }

    #[test]
    fn exploitability_surfaces_a_counter() {
        // Team 1 crushes the field (beats 2 and 3) but is hard-countered by 4.
        let reports = vec![
            report(1, 2, 3, 0),
            report(1, 3, 2, 0),
            report(4, 1, 2, 0), // 4 beats 1
        ];
        let matrix = CrossPlayMatrix::from_match_reports(&reports);
        assert!(
            matrix.mean_payoff_vs_field(1).unwrap() > 0.5,
            "1 is strong on average"
        );
        let (counter, payoff) = matrix.worst_case(1).unwrap();
        assert_eq!(counter, 4, "team 4 is the exploit");
        assert_eq!(payoff, 0.0, "1 loses to its counter every time");
    }

    #[test]
    fn unmet_pairs_have_no_payoff() {
        let matrix = CrossPlayMatrix::from_match_reports(&[report(1, 2, 1, 0)]);
        assert!(matrix.payoff(1, 3).is_none());
        assert_eq!(matrix.ids(), vec![1, 2]);
    }
}
