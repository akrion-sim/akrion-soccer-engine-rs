//! **Multimodal off-ball run prediction** (the AV "prediction head" pattern).
//!
//! [`WorldSnapshot::open_space_for`] scores a coarse 7×6 grid of candidate
//! destinations and moves the off-ball player to the single **argmax**. Two
//! problems follow from collapsing the whole scored surface to one point:
//!
//! 1. **Jitter.** The grid steps are large (6–22 yd). When two adjacent cells
//!    score near-equally, the argmax snaps between them tick-to-tick, so the
//!    target twitches across a grid step even though the player's *intent* hasn't
//!    changed. (This is exactly the granularity the `decision-refractory` work
//!    flagged it could not fix from the action side.)
//! 2. **Lost hypotheses.** A winger genuinely has two runs available — cut inside
//!    *or* hold the byline — and the scored surface is bimodal. The argmax throws
//!    the second mode away, so a *passer* or a *defender* reasoning about where
//!    this player will go has nothing but one point estimate to work with.
//!
//! Self-driving prediction stacks never collapse to a point: they emit a small
//! set of **weighted trajectory hypotheses** (multimodal), then the planner
//! reasons over the distribution. This module is that representation for soccer
//! off-ball runs.
//!
//! ## What it provides
//!
//! * [`RunHypothesis`] — one mode: a destination + its probability.
//! * [`run_hypotheses`] — greedy **spatially-distinct** mode extraction from the
//!   scored candidate surface (a softmax over the best candidate of each spatial
//!   cluster), so "cut inside" and "byline" come back as two separate weighted
//!   modes rather than being averaged into the half-space between them. This is
//!   the set a passer/defender consumes.
//! * [`blended_argmax_destination`] — the off-ball player's OWN smoothed target:
//!   a softmax blend of only the candidates in the argmax's spatial cluster, so
//!   the destination gains sub-grid resolution (killing the snap-jitter) **without**
//!   ever averaging two genuinely-distinct modes (those are deliberately excluded
//!   from the blend and survive in [`run_hypotheses`]).
//!
//! ## Parity
//!
//! Gated **off** by default ([`multimodal_run_prediction_enabled`] /
//! `DD_SOCCER_ENABLE_MULTIMODAL_RUN_PREDICTION`). When unset, `open_space_for`
//! never collects the candidate surface and keeps returning its argmax, so an
//! unconfigured process is byte-identical. All functions here are pure and
//! RNG-free, so the inspector and tests can read them without enabling the seam.

use super::*;

const MULTIMODAL_RUN_PREDICTION_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_MULTIMODAL_RUN_PREDICTION";

/// Softmax temperature (score units) over candidate scores. Larger ⇒ flatter,
/// more-shared probabilities; smaller ⇒ the top mode dominates. Tuned so two
/// candidates a couple of score units apart still share appreciable mass.
pub const RUN_PREDICTION_TEMPERATURE: f64 = 2.5;
/// Spatial radius (yards) within which two candidates count as the SAME mode /
/// cluster. Comfortably above the smallest grid step so adjacent cells fuse, well
/// below a genuine alternative run so distinct intentions stay separate.
pub const RUN_PREDICTION_CLUSTER_YARDS: f64 = 9.0;
/// Most hypotheses [`run_hypotheses`] returns (top modes by cluster-best score).
pub const RUN_PREDICTION_MAX_MODES: usize = 3;

/// Whether multimodal off-ball run prediction is consulted at the live
/// `open_space_for` chokepoint this process. Off (unset) ⇒ the argmax stands, so
/// the destination is byte-identical.
pub fn multimodal_run_prediction_enabled() -> bool {
    #[cfg(test)]
    {
        std::env::var(MULTIMODAL_RUN_PREDICTION_ENABLE_ENV).is_ok()
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var(MULTIMODAL_RUN_PREDICTION_ENABLE_ENV).is_ok())
    }
}

/// One predicted off-ball run mode: a destination and the probability the player
/// takes it, relative to the other returned modes (the returned set sums to ~1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunHypothesis {
    pub dest: Vec2,
    pub prob: f64,
}

/// Numerically-stable softmax of `scores` at `temperature`, returning weights that
/// sum to 1. Empty / all-non-finite ⇒ empty. A non-positive temperature collapses
/// to a one-hot on the max (argmax), which is the degenerate "no spread" case.
fn softmax(scores: &[f64], temperature: f64) -> Vec<f64> {
    let finite: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();
    if finite.is_empty() {
        return Vec::new();
    }
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(temperature > 0.0) {
        return scores
            .iter()
            .map(|&s| if s.is_finite() && s >= max { 1.0 } else { 0.0 })
            .collect::<Vec<_>>()
            .normalized_or_uniform();
    }
    let exps: Vec<f64> = scores
        .iter()
        .map(|&s| {
            if s.is_finite() {
                ((s - max) / temperature).exp()
            } else {
                0.0
            }
        })
        .collect();
    exps.normalized_or_uniform()
}

trait NormalizedOrUniform {
    fn normalized_or_uniform(self) -> Vec<f64>;
}
impl NormalizedOrUniform for Vec<f64> {
    fn normalized_or_uniform(self) -> Vec<f64> {
        let sum: f64 = self.iter().sum();
        let n = self.len();
        if sum > 0.0 && sum.is_finite() {
            self.into_iter().map(|w| w / sum).collect()
        } else if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            Vec::new()
        }
    }
}

/// Extract up to [`RUN_PREDICTION_MAX_MODES`] **spatially-distinct** weighted run
/// hypotheses from the scored candidate surface. Greedy farthest-point style: take
/// the highest-scoring candidate as the first mode, then repeatedly take the next
/// highest-scoring candidate at least [`RUN_PREDICTION_CLUSTER_YARDS`] from every
/// mode already chosen. The chosen modes' scores are softmaxed into probabilities.
///
/// This is the set a *passer* or *defender* reasons over — the two runs a player
/// might make, each with a likelihood — not the player's own smoothed target.
pub fn run_hypotheses(scored: &[(Vec2, f64)]) -> Vec<RunHypothesis> {
    let mut sorted: Vec<(Vec2, f64)> = scored
        .iter()
        .copied()
        .filter(|(p, s)| p.x.is_finite() && p.y.is_finite() && s.is_finite())
        .collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut modes: Vec<(Vec2, f64)> = Vec::new();
    for (p, s) in sorted {
        if modes.len() >= RUN_PREDICTION_MAX_MODES {
            break;
        }
        if modes
            .iter()
            .all(|(m, _)| m.distance(p) >= RUN_PREDICTION_CLUSTER_YARDS)
        {
            modes.push((p, s));
        }
    }
    let weights = softmax(&modes.iter().map(|(_, s)| *s).collect::<Vec<_>>(), RUN_PREDICTION_TEMPERATURE);
    modes
        .into_iter()
        .zip(weights)
        .map(|((dest, _), prob)| RunHypothesis { dest, prob })
        .collect()
}

/// The off-ball player's OWN smoothed destination: a softmax blend of only the
/// candidates within [`RUN_PREDICTION_CLUSTER_YARDS`] of the argmax, giving the
/// target sub-grid resolution so it stops snapping between adjacent grid cells.
/// Candidates outside the argmax cluster (genuinely-distinct alternative runs) are
/// excluded, so the blend never lands in the dead space between two real modes.
///
/// Returns `argmax` unchanged when there is nothing to blend (single candidate,
/// non-finite weights) so the result degrades gracefully to the old behaviour.
pub fn blended_argmax_destination(scored: &[(Vec2, f64)], argmax: Vec2) -> Vec2 {
    if !argmax.x.is_finite() || !argmax.y.is_finite() {
        return argmax;
    }
    let cluster: Vec<(Vec2, f64)> = scored
        .iter()
        .copied()
        .filter(|(p, s)| {
            p.x.is_finite()
                && p.y.is_finite()
                && s.is_finite()
                && p.distance(argmax) < RUN_PREDICTION_CLUSTER_YARDS
        })
        .collect();
    if cluster.len() <= 1 {
        return argmax;
    }
    let weights = softmax(
        &cluster.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
        RUN_PREDICTION_TEMPERATURE,
    );
    if weights.len() != cluster.len() {
        return argmax;
    }
    let mut x = 0.0;
    let mut y = 0.0;
    for ((p, _), w) in cluster.iter().zip(&weights) {
        x += p.x * w;
        y += p.y * w;
    }
    if x.is_finite() && y.is_finite() {
        Vec2::new(x, y)
    } else {
        argmax
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn gate_defaults_off() {
        let _l = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var(MULTIMODAL_RUN_PREDICTION_ENABLE_ENV);
        assert!(!multimodal_run_prediction_enabled());
    }

    #[test]
    fn softmax_is_a_normalized_distribution() {
        let w = softmax(&[1.0, 2.0, 3.0], RUN_PREDICTION_TEMPERATURE);
        assert_eq!(w.len(), 3);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // Higher score ⇒ more mass.
        assert!(w[2] > w[1] && w[1] > w[0]);
    }

    #[test]
    fn run_hypotheses_returns_spatially_distinct_modes() {
        // Two clusters far apart (cut-inside vs byline), each with a near-duplicate
        // neighbour. We must get back exactly the two distinct modes, not 4 points
        // and not an average of the clusters.
        let scored = vec![
            (Vec2::new(10.0, 50.0), 9.0),  // mode A (best)
            (Vec2::new(12.0, 51.0), 8.5),  // A's neighbour (within cluster)
            (Vec2::new(40.0, 50.0), 8.0),  // mode B (best of its cluster)
            (Vec2::new(41.0, 49.0), 7.0),  // B's neighbour
        ];
        let modes = run_hypotheses(&scored);
        assert_eq!(modes.len(), 2, "two distinct clusters ⇒ two modes");
        assert!(modes.iter().all(|m| m.prob > 0.0));
        assert!((modes.iter().map(|m| m.prob).sum::<f64>() - 1.0).abs() < 1e-9);
        // Modes are ≥ cluster radius apart.
        assert!(modes[0].dest.distance(modes[1].dest) >= RUN_PREDICTION_CLUSTER_YARDS);
        // The higher-scoring cluster (A) carries more probability.
        let a = modes.iter().min_by(|x, y| x.dest.x.total_cmp(&y.dest.x)).unwrap();
        let b = modes.iter().max_by(|x, y| x.dest.x.total_cmp(&y.dest.x)).unwrap();
        assert!(a.prob > b.prob, "A scored higher so should be likelier");
    }

    #[test]
    fn blend_smooths_within_a_cluster_only() {
        let argmax = Vec2::new(10.0, 50.0);
        // Two near-equal adjacent candidates + a far distinct one. The blend should
        // land BETWEEN the two near candidates (sub-grid smoothing) and NOT be
        // dragged toward the far mode.
        let scored = vec![
            (argmax, 9.0),
            (Vec2::new(14.0, 50.0), 8.9),
            (Vec2::new(45.0, 50.0), 8.8),
        ];
        let blended = blended_argmax_destination(&scored, argmax);
        assert!(
            blended.x > argmax.x && blended.x < 14.0,
            "blend lands between the two in-cluster cells: {blended:?}"
        );
        assert!(
            blended.x < RUN_PREDICTION_CLUSTER_YARDS + argmax.x,
            "the far mode must not drag the blend out: {blended:?}"
        );
    }

    #[test]
    fn blend_is_identity_on_a_lone_candidate() {
        let argmax = Vec2::new(20.0, 30.0);
        assert_eq!(
            blended_argmax_destination(&[(argmax, 5.0)], argmax),
            argmax,
            "nothing to blend ⇒ the argmax is returned unchanged"
        );
    }

    #[test]
    fn handles_non_finite_gracefully() {
        let argmax = Vec2::new(f64::NAN, 1.0);
        assert!(blended_argmax_destination(&[(argmax, 1.0)], argmax).x.is_nan());
        assert!(run_hypotheses(&[(Vec2::new(f64::INFINITY, 0.0), f64::NAN)]).is_empty());
    }
}
