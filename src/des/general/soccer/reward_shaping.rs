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

/// Env gate for the reward shaping-**discipline** layer: discount-aligned PBRS
/// ([`disciplined_gamma`]) and a per-step dense-shaping budget
/// ([`apply_dense_shaping_budget`]). OFF (default) ⇒ `γ` folds to `1.0` and the
/// budget is inert, so the per-step reward is byte-identical. Turning it on is
/// behaviour-changing for the trained policy (retrain / A/B against the current
/// learner — measure Elo / win-rate, not raw reward); see
/// `docs/reward-shaping-audit.md` follow-ups #1 and #3.
const SHAPING_DISCIPLINE_ENABLE_ENV: &str = "DD_SOCCER_ENABLE_SHAPING_DISCIPLINE";

fn shaping_discipline_env() -> bool {
    std::env::var(SHAPING_DISCIPLINE_ENABLE_ENV)
        .map(|raw| {
            let v = raw.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Whether the reward shaping-discipline layer is active this process. Tests
/// re-read the env each call (so an A/B can toggle it); production caches once.
pub fn shaping_discipline_enabled() -> bool {
    #[cfg(test)]
    {
        shaping_discipline_env()
    }
    #[cfg(not(test))]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        // Promoted to default-ON in production (reward shaping-discipline; training-time only).
        *ENABLED.get_or_init(|| super::gate_default_on(SHAPING_DISCIPLINE_ENABLE_ENV))
    }
}

/// The discount PBRS shaping should use to be policy-invariant: the learner's
/// [`REWARD_SHAPING_DEFAULT_GAMMA`] when the discipline is on, else `1.0` — the
/// historical implicit discount of the plain `Φ(after) − Φ(before)` delta terms.
/// PBRS is only invariant when this matches the discount the returns/advantages
/// use, so wiring shaping callers through it closes the γ=1-vs-0.99 gap (audit
/// follow-up #1) behind the discipline gate.
pub fn disciplined_gamma() -> f64 {
    if shaping_discipline_enabled() {
        REWARD_SHAPING_DEFAULT_GAMMA
    } else {
        1.0
    }
}

/// Clamp a per-step **dense** shaping reward to `±budget` so the accreted dense
/// shards cannot, on a single step, dominate the sparse match-outcome signal
/// (goals are summed separately downstream and stay uncapped — see
/// `learning_transitions_for`). The audit's follow-up #3 "single shaping budget".
/// Inert (returns `reward` unchanged) when the discipline is off or `budget` is
/// non-finite / non-positive, so the default process is byte-identical; a
/// non-finite `reward` is neutralised to `0.0` so a degenerate shard can't inject
/// a `NaN`/`inf` into a gradient.
pub fn apply_dense_shaping_budget(reward: f64, budget: f64) -> f64 {
    if !reward.is_finite() {
        return 0.0;
    }
    if !shaping_discipline_enabled() || !budget.is_finite() || budget <= 0.0 {
        return reward;
    }
    reward.clamp(-budget, budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The discipline gate reads a process-global env var; serialize the tests that
    // toggle it so they can't race each other under the parallel test runner.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_discipline<R>(f: impl FnOnce() -> R) -> R {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(SHAPING_DISCIPLINE_ENABLE_ENV, "1");
        let r = f();
        std::env::remove_var(SHAPING_DISCIPLINE_ENABLE_ENV);
        r
    }

    #[test]
    fn discipline_gate_default_off_is_identity() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var(SHAPING_DISCIPLINE_ENABLE_ENV);
        assert!(!shaping_discipline_enabled());
        assert_eq!(disciplined_gamma(), 1.0);
        // Budget inert when off ⇒ reward passes through unchanged.
        assert_eq!(apply_dense_shaping_budget(999.0, 4.0), 999.0);
    }

    #[test]
    fn disciplined_gamma_matches_the_learner_discount_when_on() {
        with_discipline(|| {
            assert_eq!(disciplined_gamma(), REWARD_SHAPING_DEFAULT_GAMMA);
            // A held potential now telescopes to a small time cost instead of 0.
            let hold = potential_based_shaping(disciplined_gamma(), 0.09, 0.09);
            assert!(
                hold < 0.0 && hold > -0.01,
                "holding a state has a tiny γ time-cost: {hold}"
            );
        });
    }

    #[test]
    fn budget_caps_dense_shaping_only_when_on() {
        with_discipline(|| {
            assert_eq!(apply_dense_shaping_budget(50.0, 6.0), 6.0);
            assert_eq!(apply_dense_shaping_budget(-50.0, 6.0), -6.0);
            assert_eq!(
                apply_dense_shaping_budget(2.0, 6.0),
                2.0,
                "within budget ⇒ untouched"
            );
            // A non-positive / non-finite budget disables the cap (still on).
            assert_eq!(apply_dense_shaping_budget(50.0, 0.0), 50.0);
            assert_eq!(apply_dense_shaping_budget(50.0, f64::INFINITY), 50.0);
        });
        // Non-finite reward is always neutralised, gate or no gate.
        assert_eq!(apply_dense_shaping_budget(f64::NAN, 6.0), 0.0);
    }

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
        assert!(
            total.abs() < 1e-12,
            "returning trajectory must net to zero, got {total}"
        );
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
