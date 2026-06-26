//! Potential-based reward shaping (PBRS) — the disciplined way to add dense
//! guidance to the learner without biasing the optimal policy.
//!
//! Ng, Harada & Russell (1999) proved that a shaping reward of the form
//!
//! ```text
//!     F(s, s') = γ · Φ(s') − Φ(s)
//! ```
//!
//! leaves the set of optimal policies unchanged for *any* real-valued potential
//! `Φ` over states, under the discounted-return objective with discount `γ`.
//! The intuition: over a trajectory the `−Φ(s)` of one step cancels the
//! `γΦ(s')` carried from the previous step, so the shaping contributes only a
//! telescoping `−Φ(s₀)` (a constant) plus, for an absorbing terminal, a
//! `γᵀ Φ(s_T)` boundary term. It moves *where* reward arrives (earlier, denser)
//! without moving *what* the agent is ultimately optimising.
//!
//! Why it matters here: the soccer reward has accreted many dense shards. Those
//! expressed as a state *difference* (`f(after) − f(before)` — pitch value,
//! yards-to-goal progress, ball-forward) are already potential-shaped and safe.
//! Those expressed as a *flat* state-conditioned bonus (`+0.09` for being in a
//! zone, a fixed penalty for holding) are NOT: they add a per-visit bias the
//! policy can farm, distorting the objective away from "win the match". New
//! state-conditioned shaping should route through [`potential_based_shaping`]
//! so it is invariant by construction; see `docs/reward-shaping-audit.md`.

/// The learner's default discount. PBRS is only policy-invariant when the `γ`
/// used here matches the discount the returns/advantages are computed with, so
/// shaping callers should pass the same `γ` their value target uses. Exposed so
/// shaping terms and the critic cannot silently disagree.
pub const REWARD_SHAPING_DEFAULT_GAMMA: f64 = 0.99;

/// Potential-based shaping reward `F = γ·Φ(s') − Φ(s)` (Ng et al. 1999).
///
/// `gamma` is the discount the value target uses; `potential_before` /
/// `potential_after` are `Φ(s)` and `Φ(s')`. Non-finite inputs (or a
/// non-finite result) yield `0.0` so a degenerate potential can never inject a
/// `NaN`/`inf` into a gradient. With `gamma == 1.0` this is the plain state
/// difference `Φ(s') − Φ(s)` many existing terms already use.
pub fn potential_based_shaping(gamma: f64, potential_before: f64, potential_after: f64) -> f64 {
    if !gamma.is_finite() || !potential_before.is_finite() || !potential_after.is_finite() {
        return 0.0;
    }
    let shaped = gamma * potential_after - potential_before;
    if shaped.is_finite() {
        shaped
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_one_is_the_plain_state_difference() {
        // The form many existing delta-rewards already use.
        assert_eq!(potential_based_shaping(1.0, 0.3, 0.5), 0.2_f64);
        assert_eq!(potential_based_shaping(1.0, 0.5, 0.5), 0.0);
    }

    #[test]
    fn discounted_form_matches_the_ng_definition() {
        let f = potential_based_shaping(0.9, 1.0, 2.0);
        assert!((f - (0.9 * 2.0 - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn telescopes_to_a_constant_over_a_returning_trajectory() {
        // Key invariance property: for γ = 1, the shaping over a trajectory that
        // returns to its starting potential sums to zero — it adds no net reward,
        // only redistributes it across time. So it cannot change which policy is
        // optimal, only how fast the signal arrives.
        let phis = [0.2_f64, 0.7, 0.4, 0.9, 0.2];
        let total: f64 = phis
            .windows(2)
            .map(|w| potential_based_shaping(1.0, w[0], w[1]))
            .sum();
        assert!(total.abs() < 1e-12, "returning trajectory must net to zero, got {total}");
    }

    #[test]
    fn non_finite_inputs_are_neutralised() {
        assert_eq!(potential_based_shaping(f64::NAN, 0.1, 0.2), 0.0);
        assert_eq!(potential_based_shaping(1.0, f64::INFINITY, 0.2), 0.0);
        assert_eq!(potential_based_shaping(1.0, 0.1, f64::NAN), 0.0);
        // A finite computation that overflows to inf is also caught.
        assert_eq!(potential_based_shaping(f64::MAX, 1.0, f64::MAX), 0.0);
    }
}
