//! Hermetic DEN-862 plateau-evidence contract.
//!
//! This intentionally does not claim that repeated action selection means a policy is solved.
//! A plateau requires held-out stagnation plus corroborating evidence, and all inputs are
//! validated before a decision is emitted.

#[derive(Clone, Copy, Debug)]
struct Window {
    held_out_score: f64,
    policy_entropy: f64,
    action_diversity: f64,
    state_coverage: f64,
    value_error: f64,
}

#[derive(Debug, PartialEq, Eq)]
enum Decision {
    ContinueLearning,
    PlateauSuspected { reasons: Vec<&'static str> },
}

#[derive(Clone, Copy, Debug)]
struct Thresholds {
    max_abs_score_change: f64,
    max_entropy: f64,
    max_action_diversity: f64,
    max_coverage_gain: f64,
    min_value_error: f64,
}

fn validate_window(window: Window) -> Result<(), &'static str> {
    let values = [
        window.held_out_score,
        window.policy_entropy,
        window.action_diversity,
        window.state_coverage,
        window.value_error,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("metrics must be finite");
    }
    if !(0.0..=1.0).contains(&window.policy_entropy) {
        return Err("policy entropy must be normalized to [0,1]");
    }
    if !(0.0..=1.0).contains(&window.action_diversity) {
        return Err("action diversity must be normalized to [0,1]");
    }
    if !(0.0..=1.0).contains(&window.state_coverage) {
        return Err("state coverage must be normalized to [0,1]");
    }
    if window.value_error < 0.0 {
        return Err("value error cannot be negative");
    }
    Ok(())
}

fn assess_plateau(
    previous: Window,
    current: Window,
    thresholds: Thresholds,
) -> Result<Decision, &'static str> {
    validate_window(previous)?;
    validate_window(current)?;

    let score_change = (current.held_out_score - previous.held_out_score).abs();
    if score_change > thresholds.max_abs_score_change {
        return Ok(Decision::ContinueLearning);
    }

    let coverage_gain = current.state_coverage - previous.state_coverage;
    if coverage_gain > thresholds.max_coverage_gain {
        return Ok(Decision::ContinueLearning);
    }

    let mut reasons = Vec::new();
    if current.policy_entropy <= thresholds.max_entropy {
        reasons.push("low_policy_entropy");
    }
    if current.action_diversity <= thresholds.max_action_diversity {
        reasons.push("low_action_diversity");
    }
    if current.value_error >= thresholds.min_value_error {
        reasons.push("held_out_value_error_persists");
    }

    // Held-out stagnation by itself is deliberately insufficient. At least two independent
    // corroborating signals are required before exploration/diagnostic policy may escalate.
    if reasons.len() >= 2 {
        Ok(Decision::PlateauSuspected { reasons })
    } else {
        Ok(Decision::ContinueLearning)
    }
}

fn defaults() -> Thresholds {
    Thresholds {
        max_abs_score_change: 0.005,
        max_entropy: 0.08,
        max_action_diversity: 0.12,
        max_coverage_gain: 0.002,
        min_value_error: 0.05,
    }
}

fn main() {
    let previous = Window {
        held_out_score: 0.742,
        policy_entropy: 0.09,
        action_diversity: 0.14,
        state_coverage: 0.611,
        value_error: 0.071,
    };
    let current = Window {
        held_out_score: 0.743,
        policy_entropy: 0.05,
        action_diversity: 0.08,
        state_coverage: 0.6115,
        value_error: 0.068,
    };

    let decision = assess_plateau(previous, current, defaults()).expect("valid fixture");
    println!("DEN-862 plateau evidence: {decision:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Window {
        Window {
            held_out_score: 0.70,
            policy_entropy: 0.20,
            action_diversity: 0.30,
            state_coverage: 0.50,
            value_error: 0.08,
        }
    }

    #[test]
    fn repeated_behavior_is_not_proof_of_plateau_when_held_out_score_improves() {
        let previous = base();
        let current = Window {
            held_out_score: 0.73,
            policy_entropy: 0.02,
            action_diversity: 0.02,
            state_coverage: 0.50,
            value_error: 0.08,
        };
        assert_eq!(
            assess_plateau(previous, current, defaults()).unwrap(),
            Decision::ContinueLearning
        );
    }

    #[test]
    fn coverage_growth_prevents_false_plateau_classification() {
        let previous = base();
        let current = Window {
            held_out_score: 0.701,
            policy_entropy: 0.02,
            action_diversity: 0.02,
            state_coverage: 0.54,
            value_error: 0.08,
        };
        assert_eq!(
            assess_plateau(previous, current, defaults()).unwrap(),
            Decision::ContinueLearning
        );
    }

    #[test]
    fn stagnation_plus_multiple_corroborating_signals_flags_plateau() {
        let previous = base();
        let current = Window {
            held_out_score: 0.701,
            policy_entropy: 0.04,
            action_diversity: 0.06,
            state_coverage: 0.501,
            value_error: 0.09,
        };
        let decision = assess_plateau(previous, current, defaults()).unwrap();
        assert_eq!(
            decision,
            Decision::PlateauSuspected {
                reasons: vec![
                    "low_policy_entropy",
                    "low_action_diversity",
                    "held_out_value_error_persists",
                ],
            }
        );
    }

    #[test]
    fn held_out_stagnation_alone_is_not_enough() {
        let previous = base();
        let current = Window {
            held_out_score: 0.701,
            policy_entropy: 0.30,
            action_diversity: 0.40,
            state_coverage: 0.501,
            value_error: 0.01,
        };
        assert_eq!(
            assess_plateau(previous, current, defaults()).unwrap(),
            Decision::ContinueLearning
        );
    }

    #[test]
    fn invalid_or_nonfinite_metrics_fail_closed() {
        let previous = base();
        let invalid = Window {
            policy_entropy: f64::NAN,
            ..base()
        };
        assert_eq!(
            assess_plateau(previous, invalid, defaults()).unwrap_err(),
            "metrics must be finite"
        );

        let invalid = Window {
            value_error: -0.1,
            ..base()
        };
        assert_eq!(
            assess_plateau(previous, invalid, defaults()).unwrap_err(),
            "value error cannot be negative"
        );
    }
}
