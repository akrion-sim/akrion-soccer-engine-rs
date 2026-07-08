//! Promotion gate: decide whether a *candidate* brain is actually better than the
//! incumbent before it replaces it in the live/league pool.
//!
//! Raw training reward is a training instrument, not the truth — reward shaping
//! changes, reward-hacking, and a drifting opponent field all make "my reward went
//! up" an unreliable promotion signal (the whole reason [`crate::des::general::soccer_elo`]
//! exists). The truth is: *does this brain beat a diverse, frozen field on matches it
//! never trained on?* This module folds a set of **held-out** fixtures (seeds disjoint
//! from training — the caller owns that disjointness; see the eval-gate runner) into a
//! single promote/reject verdict built from three orthogonal reads:
//!
//!   * **Strength** — mean cross-play payoff vs the frozen field
//!     ([`CrossPlayMatrix::mean_payoff_vs_field`]) and the held-out Elo delta vs the
//!     baseline. "Is it better on average?"
//!   * **Confidence** — a Wilson score lower bound on that mean, so a lucky 3-game
//!     edge does not promote. "Is the edge real, or noise?"
//!   * **Robustness** — worst-case single-opponent payoff
//!     ([`CrossPlayMatrix::worst_case`]). A brain that crushes the field on average
//!     but is hard-countered by one frozen opponent is *different, not better*, and
//!     promoting it invites a cycle. "Is it exploitable?"
//!
//! Promote iff all three clear their thresholds. Pure + deterministic given the
//! fixtures: no engine, RNG, or I/O. The runner binary supplies the held-out
//! `MatchReport`s by playing the candidate against the frozen pool with a frozen
//! [`crate::des::general::tournament::EngineMatchRunner`].

use crate::des::general::soccer_elo::{CrossPlayMatrix, EloRatings, ELO_DEFAULT_K};
use crate::des::general::tournament::MatchReport;

/// Default minimum held-out games vs the field below which the gate refuses to
/// decide (auto-reject): too little evidence to promote on.
pub const PROMOTION_MIN_GAMES: u32 = 8;

/// Default mean-payoff bar the candidate's Wilson lower bound must clear vs the
/// field. `0.5` = "better than break-even with 95% confidence". Anything ≤ 0.5
/// means we are not confident it beats the field at all.
pub const PROMOTION_WILSON_FLOOR: f64 = 0.5;

/// Default exploitability floor: the candidate's worst single-opponent payoff must
/// be at least this. `0.35` tolerates losing a matchup on balance but rejects a
/// hard counter (a frozen opponent that beats it ≥ ~65% of the time).
pub const PROMOTION_WORST_CASE_FLOOR: f64 = 0.35;

/// Default z for the Wilson interval (95% one-sided ≈ two-sided 90%; 1.96 is the
/// common two-sided-95% z and the conservative default here).
pub const PROMOTION_WILSON_Z: f64 = 1.96;

/// Wilson score-interval **lower** bound for an observed mean score `p_hat` over
/// `n` games at confidence `z`. We treat a match score in `[0, 1]` (win=1,
/// draw=0.5, loss=0) as a proportion-like statistic — an approximation, since a
/// run of draws is lower-variance than the Bernoulli the interval assumes, so the
/// bound is *conservative* (it under-states confidence for draw-heavy fields),
/// which is the safe direction for a promotion gate. Returns `0.0` for `n == 0`.
pub fn wilson_lower_bound(p_hat: f64, n: u32, z: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let p = p_hat.clamp(0.0, 1.0);
    let n = n as f64;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let margin = (z / denom) * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt();
    (center - margin).clamp(0.0, 1.0)
}

/// Draw-aware lower confidence bound on the mean match score, using the TRUE empirical variance
/// of the `{1, 0.5, 0}` score distribution instead of the Bernoulli `p(1-p)` the Wilson interval
/// assumes. A draw-heavy record concentrates mass at 0.5, so its per-game variance is up to ~2×
/// smaller than Bernoulli — the Wilson bound is then over-conservative and REJECTS genuine
/// draw-heavy climbs (the measurement-plateau bug: a confirmed +Elo/+GD candidate at mean 0.56
/// fails the gate because `p(1-p)` overstates its variance). This normal lower bound
/// `mean − z·sqrt(var/n)` matches the actual evidence. Returns `0.0` for `n == 0`.
pub fn draw_aware_lower_bound(record: &CandidateRecord, z: f64) -> f64 {
    let n = record.games();
    if n == 0 {
        return 0.0;
    }
    let n_f = f64::from(n);
    let mean = record.mean_score();
    // E[X²] for X ∈ {1, 0.5, 0}: wins contribute 1, draws 0.25, losses 0.
    let e_x2 = (f64::from(record.wins) + 0.25 * f64::from(record.draws)) / n_f;
    let var = (e_x2 - mean * mean).max(0.0);
    let se = (var / n_f).sqrt();
    (mean - z * se).clamp(0.0, 1.0)
}

/// Whether the promotion gate uses the draw-aware empirical-variance lower bound instead of the
/// Bernoulli Wilson bound (`DD_SOCCER_ENABLE_DRAW_AWARE_WILSON`). Default-off ⇒ byte-identical;
/// enable it in the anti-plateau config so genuine draw-heavy climbs actually promote instead of
/// being rejected by the over-conservative Bernoulli variance.
pub fn draw_aware_wilson_enabled() -> bool {
    std::env::var("DD_SOCCER_ENABLE_DRAW_AWARE_WILSON")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// The candidate's held-out record against the whole frozen field (every fixture
/// where it played, either orientation).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CandidateRecord {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub goals_for: u32,
    pub goals_against: u32,
}

impl CandidateRecord {
    pub fn games(&self) -> u32 {
        self.wins + self.draws + self.losses
    }

    /// Mean match score in `[0, 1]` (win=1, draw=0.5, loss=0) — the field-average
    /// strength the Wilson bound is taken on.
    pub fn mean_score(&self) -> f64 {
        let games = self.games();
        if games == 0 {
            return 0.0;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / games as f64
    }

    pub fn goal_difference(&self) -> i32 {
        self.goals_for as i32 - self.goals_against as i32
    }
}

/// Tunable thresholds for [`evaluate_promotion`]. Defaults are the `PROMOTION_*`
/// constants; override per call site (a high-stakes live promotion can demand more
/// games / a higher Wilson floor than a routine league refresh).
#[derive(Clone, Copy, Debug)]
pub struct PromotionThresholds {
    pub min_games: u32,
    pub wilson_floor: f64,
    pub worst_case_floor: f64,
    pub wilson_z: f64,
    pub elo_k: f64,
}

impl Default for PromotionThresholds {
    fn default() -> Self {
        PromotionThresholds {
            min_games: PROMOTION_MIN_GAMES,
            wilson_floor: PROMOTION_WILSON_FLOOR,
            worst_case_floor: PROMOTION_WORST_CASE_FLOOR,
            wilson_z: PROMOTION_WILSON_Z,
            elo_k: ELO_DEFAULT_K,
        }
    }
}

/// The gate's decision plus every metric it weighed, so an operator (or the log)
/// can see *why* — not just the boolean. All payoffs are in `[0, 1]` from the
/// candidate's perspective; Elo deltas are folded fresh from the held-out fixtures
/// (candidate and baseline both anchored at 1500), so the delta is a clean
/// held-out read, not contaminated by pre-match ratings.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionVerdict {
    pub promote: bool,
    /// Candidate Elo from the held-out fold.
    pub candidate_elo: f64,
    /// Baseline (incumbent) Elo from the held-out fold.
    pub baseline_elo: f64,
    /// `candidate_elo - baseline_elo` over the held-out fixtures.
    pub elo_delta: f64,
    /// Mean cross-play payoff vs the whole frozen field (`None` if the candidate
    /// never met anyone — also an auto-reject).
    pub mean_payoff_vs_field: Option<f64>,
    /// Direct payoff vs the named baseline (`None` if they never met).
    pub payoff_vs_baseline: Option<f64>,
    /// Worst-case `(opponent_id, payoff)` — the candidate's most exploitative
    /// matchup. `None` if it played no one.
    pub worst_case: Option<(usize, f64)>,
    /// Wilson lower bound on [`CandidateRecord::mean_score`].
    pub wilson_lower_bound: f64,
    pub record: CandidateRecord,
    /// Human-readable reasons (every failed check, or the passing summary).
    pub reasons: Vec<String>,
}

/// Tally the candidate's held-out record against the field from the raw fixtures.
fn candidate_record(reports: &[MatchReport], candidate_id: usize) -> CandidateRecord {
    let mut record = CandidateRecord::default();
    for report in reports {
        let (gf, ga, won_shootout) = if report.home_id == candidate_id {
            (
                report.home_goals,
                report.away_goals,
                report.shootout_winner.map(|w| w == candidate_id),
            )
        } else if report.away_id == candidate_id {
            (
                report.away_goals,
                report.home_goals,
                report.shootout_winner.map(|w| w == candidate_id),
            )
        } else {
            continue;
        };
        record.goals_for += gf;
        record.goals_against += ga;
        match gf.cmp(&ga) {
            std::cmp::Ordering::Greater => record.wins += 1,
            std::cmp::Ordering::Less => record.losses += 1,
            std::cmp::Ordering::Equal => match won_shootout {
                // A knockout-style held-out fixture resolved on a shootout counts
                // for whoever it was awarded to (mirrors `soccer_elo::match_score`).
                Some(true) => record.wins += 1,
                Some(false) => record.losses += 1,
                None => record.draws += 1,
            },
        }
    }
    record
}

/// Decide whether `candidate_id` should be promoted over `baseline_id` given a set
/// of **held-out** `reports` in which the candidate played the frozen field
/// (baseline included). The reports must be from seeds disjoint from training for
/// the verdict to mean anything; this function does not (cannot) verify that.
pub fn evaluate_promotion(
    reports: &[MatchReport],
    candidate_id: usize,
    baseline_id: usize,
    thresholds: PromotionThresholds,
) -> PromotionVerdict {
    let elo = EloRatings::from_match_reports(reports, thresholds.elo_k, 12);
    let matrix = CrossPlayMatrix::from_match_reports(reports);
    let record = candidate_record(reports, candidate_id);

    let candidate_elo = elo.rating(candidate_id);
    let baseline_elo = elo.rating(baseline_id);
    let elo_delta = candidate_elo - baseline_elo;
    let mean_payoff_vs_field = matrix.mean_payoff_vs_field(candidate_id);
    let payoff_vs_baseline = matrix.payoff(candidate_id, baseline_id);
    let worst_case = matrix.worst_case(candidate_id);
    let wilson = wilson_lower_bound(record.mean_score(), record.games(), thresholds.wilson_z);

    let mut reasons = Vec::new();
    let mut promote = true;

    if record.games() < thresholds.min_games {
        promote = false;
        reasons.push(format!(
            "insufficient evidence: {} held-out games < min {}",
            record.games(),
            thresholds.min_games
        ));
    }
    if mean_payoff_vs_field.is_none() {
        promote = false;
        reasons.push("candidate never met the field (no cross-play payoff)".to_string());
    }
    if wilson < thresholds.wilson_floor {
        promote = false;
        reasons.push(format!(
            "not confident vs field: Wilson lower bound {:.3} < floor {:.3} (mean score {:.3} over {} games)",
            wilson,
            thresholds.wilson_floor,
            record.mean_score(),
            record.games()
        ));
    }
    match worst_case {
        Some((opp, payoff)) if payoff < thresholds.worst_case_floor => {
            promote = false;
            reasons.push(format!(
                "exploitable: worst-case payoff {payoff:.3} vs opponent {opp} < floor {:.3}",
                thresholds.worst_case_floor
            ));
        }
        _ => {}
    }

    if promote {
        reasons.push(format!(
            "PROMOTE: Wilson {wilson:.3} ≥ {:.3}, Elo Δ {elo_delta:+.1}, mean payoff {:.3}, worst-case {:.3} over {} games",
            thresholds.wilson_floor,
            mean_payoff_vs_field.unwrap_or(0.0),
            worst_case.map(|(_, p)| p).unwrap_or(0.0),
            record.games(),
        ));
    }

    PromotionVerdict {
        promote,
        candidate_elo,
        baseline_elo,
        elo_delta,
        mean_payoff_vs_field,
        payoff_vs_baseline,
        worst_case,
        wilson_lower_bound: wilson,
        record,
        reasons,
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

    /// A held-out slate where the candidate (1) reliably beats a 4-team frozen
    /// field (2,3,4,5 = baseline + three others), twice each (8 games, clearing the
    /// evidence floor), with no hard counter.
    fn dominant_slate() -> Vec<MatchReport> {
        let mut reports = Vec::new();
        for &opp in &[2usize, 3, 4, 5] {
            reports.push(report(1, opp, 2, 0));
            reports.push(report(opp, 1, 0, 1)); // candidate wins away too
        }
        reports
    }

    #[test]
    fn wilson_lower_bound_is_conservative_and_grows_with_n() {
        // 100% over few games is NOT confidently > 0.5; over many games it is.
        let few = wilson_lower_bound(1.0, 3, PROMOTION_WILSON_Z);
        let many = wilson_lower_bound(1.0, 50, PROMOTION_WILSON_Z);
        assert!(
            few < many,
            "more games ⇒ tighter (higher) lower bound: {few} vs {many}"
        );
        assert!(few < 1.0 && few > 0.0);
        // n == 0 is the floor.
        assert_eq!(wilson_lower_bound(1.0, 0, PROMOTION_WILSON_Z), 0.0);
        // A coin-flip mean never clears 0.5 from below.
        assert!(wilson_lower_bound(0.5, 40, PROMOTION_WILSON_Z) < 0.5);
    }

    #[test]
    fn candidate_record_counts_both_orientations_and_shootouts() {
        let reports = vec![
            report(1, 2, 3, 1), // home win
            report(2, 1, 0, 2), // away win
            report(1, 3, 1, 1), // draw
            MatchReport {
                // level, shootout to candidate
                shootout_winner: Some(1),
                ..report(1, 4, 2, 2)
            },
        ];
        let rec = candidate_record(&reports, 1);
        assert_eq!(rec.wins, 3);
        assert_eq!(rec.draws, 1);
        assert_eq!(rec.losses, 0);
        assert_eq!(rec.goals_for, 3 + 2 + 1 + 2);
        assert_eq!(rec.goals_against, 1 + 0 + 1 + 2);
        assert!((rec.mean_score() - (3.0 + 0.5) / 4.0).abs() < 1e-9);
    }

    #[test]
    fn dominant_candidate_is_promoted() {
        let verdict = evaluate_promotion(&dominant_slate(), 1, 2, PromotionThresholds::default());
        assert!(verdict.promote, "reasons: {:?}", verdict.reasons);
        assert!(verdict.elo_delta > 0.0);
        assert!(verdict.mean_payoff_vs_field.unwrap() > 0.5);
        assert!(verdict.wilson_lower_bound >= PROMOTION_WILSON_FLOOR);
    }

    #[test]
    fn even_candidate_is_rejected_for_lack_of_confidence() {
        // Splits every series 1-1 ⇒ mean score 0.5 ⇒ Wilson lower bound < 0.5.
        let mut reports = Vec::new();
        for &opp in &[2usize, 3, 4] {
            reports.push(report(1, opp, 1, 0));
            reports.push(report(1, opp, 0, 1));
        }
        let verdict = evaluate_promotion(&reports, 1, 2, PromotionThresholds::default());
        assert!(!verdict.promote);
        assert!(verdict.reasons.iter().any(|r| r.contains("not confident")));
    }

    #[test]
    fn exploitable_candidate_is_rejected_even_if_strong_on_average() {
        // Candidate crushes 2 and 3 but is hard-countered by 4 (loses every time).
        let mut reports = Vec::new();
        for &opp in &[2usize, 3] {
            for _ in 0..4 {
                reports.push(report(1, opp, 3, 0));
            }
        }
        for _ in 0..4 {
            reports.push(report(4, 1, 2, 0)); // 4 beats the candidate every time
        }
        let verdict = evaluate_promotion(&reports, 1, 2, PromotionThresholds::default());
        assert!(
            !verdict.promote,
            "a hard-countered brain is different, not better"
        );
        assert_eq!(verdict.worst_case.map(|(o, _)| o), Some(4));
        assert!(verdict.reasons.iter().any(|r| r.contains("exploitable")));
    }

    #[test]
    fn too_few_games_auto_rejects() {
        let reports = vec![report(1, 2, 5, 0), report(1, 3, 4, 0)];
        let verdict = evaluate_promotion(&reports, 1, 2, PromotionThresholds::default());
        assert!(!verdict.promote);
        assert!(verdict
            .reasons
            .iter()
            .any(|r| r.contains("insufficient evidence")));
    }

    #[test]
    fn candidate_that_never_played_is_rejected() {
        let reports = vec![report(2, 3, 1, 0), report(3, 4, 2, 1)];
        let verdict = evaluate_promotion(&reports, 1, 2, PromotionThresholds::default());
        assert!(!verdict.promote);
        assert!(verdict.mean_payoff_vs_field.is_none());
        assert_eq!(verdict.record.games(), 0);
    }
}
