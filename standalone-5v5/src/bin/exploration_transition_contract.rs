//! Replay-safe transition evidence for DEN-862 exploration experiments.
//!
//! This complements `exploration_contract.rs`: action selection owns the
//! behavior distribution, while this contract proves the transition payload
//! retains enough evidence for off-policy/on-policy learning and replay.

const SCHEMA_VERSION: u16 = 1;
const LOG_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExplorationReason {
    Greedy,
    Epsilon,
    RankWeighted,
    Boltzmann,
    Uncertainty,
}

impl ExplorationReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Epsilon => "epsilon",
            Self::RankWeighted => "rank_weighted",
            Self::Boltzmann => "boltzmann",
            Self::Uncertainty => "uncertainty",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CandidateEvidence {
    action_id: u32,
    rank: usize,
    estimated_value: f64,
    uncertainty: f64,
    feasible: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct TransitionEvidence {
    schema_version: u16,
    seed: u64,
    decision_nonce: u64,
    selected_action_id: u32,
    selected_rank: usize,
    estimated_value: f64,
    uncertainty: f64,
    behavior_probability: f64,
    behavior_log_probability: f64,
    exploration_reason: ExplorationReason,
    feasible_action_ids: Vec<u32>,
}

impl TransitionEvidence {
    fn from_selection(
        seed: u64,
        decision_nonce: u64,
        selected_action_id: u32,
        behavior_probability: f64,
        reason: ExplorationReason,
        candidates: &[CandidateEvidence],
    ) -> Result<Self, String> {
        if !behavior_probability.is_finite()
            || behavior_probability <= 0.0
            || behavior_probability > 1.0
        {
            return Err("behavior probability must be finite and in (0, 1]".into());
        }

        let selected = candidates
            .iter()
            .find(|candidate| candidate.action_id == selected_action_id)
            .ok_or_else(|| format!("selected action {selected_action_id} is absent"))?;
        if !selected.feasible {
            return Err(format!("selected action {selected_action_id} is infeasible"));
        }
        if !selected.estimated_value.is_finite() {
            return Err("selected estimated value must be finite".into());
        }
        if !selected.uncertainty.is_finite() || selected.uncertainty < 0.0 {
            return Err("selected uncertainty must be finite and non-negative".into());
        }

        let mut feasible: Vec<_> = candidates.iter().filter(|candidate| candidate.feasible).collect();
        feasible.sort_by_key(|candidate| candidate.rank);
        if feasible
            .iter()
            .enumerate()
            .any(|(expected_rank, candidate)| candidate.rank != expected_rank)
        {
            return Err("feasible ranks must be contiguous from zero".into());
        }

        Ok(Self {
            schema_version: SCHEMA_VERSION,
            seed,
            decision_nonce,
            selected_action_id,
            selected_rank: selected.rank,
            estimated_value: selected.estimated_value,
            uncertainty: selected.uncertainty,
            behavior_probability,
            behavior_log_probability: behavior_probability.ln(),
            exploration_reason: reason,
            feasible_action_ids: feasible.iter().map(|candidate| candidate.action_id).collect(),
        })
    }

    fn validate_for_learning(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err("unsupported transition evidence schema".into());
        }
        if !self.feasible_action_ids.contains(&self.selected_action_id) {
            return Err("selected action missing from feasibility mask".into());
        }
        if self.feasible_action_ids.get(self.selected_rank) != Some(&self.selected_action_id) {
            return Err("selected rank disagrees with feasibility ordering".into());
        }
        if !self.behavior_probability.is_finite()
            || self.behavior_probability <= 0.0
            || self.behavior_probability > 1.0
        {
            return Err("invalid behavior probability".into());
        }
        let expected_log = self.behavior_probability.ln();
        if !self.behavior_log_probability.is_finite()
            || (self.behavior_log_probability - expected_log).abs() > LOG_TOLERANCE
        {
            return Err("behavior log probability does not match sampling probability".into());
        }
        if !self.estimated_value.is_finite() {
            return Err("estimated value must be finite".into());
        }
        if !self.uncertainty.is_finite() || self.uncertainty < 0.0 {
            return Err("uncertainty must be finite and non-negative".into());
        }
        Ok(())
    }

    fn to_json(&self) -> String {
        let feasible = self
            .feasible_action_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"schemaVersion\":{},\"seed\":{},\"decisionNonce\":{},",
                "\"selectedActionId\":{},\"selectedRank\":{},",
                "\"estimatedValue\":{},\"uncertainty\":{},",
                "\"behaviorProbability\":{},\"behaviorLogProbability\":{},",
                "\"explorationReason\":\"{}\",\"feasibleActionIds\":[{}]}}"
            ),
            self.schema_version,
            self.seed,
            self.decision_nonce,
            self.selected_action_id,
            self.selected_rank,
            self.estimated_value,
            self.uncertainty,
            self.behavior_probability,
            self.behavior_log_probability,
            self.exploration_reason.as_str(),
            feasible,
        )
    }
}

fn fixture() -> [CandidateEvidence; 4] {
    [
        CandidateEvidence {
            action_id: 10,
            rank: 0,
            estimated_value: 5.0,
            uncertainty: 0.1,
            feasible: true,
        },
        CandidateEvidence {
            action_id: 20,
            rank: 1,
            estimated_value: 4.0,
            uncertainty: 0.2,
            feasible: true,
        },
        CandidateEvidence {
            action_id: 30,
            rank: 2,
            estimated_value: 3.0,
            uncertainty: 2.0,
            feasible: true,
        },
        CandidateEvidence {
            action_id: 40,
            rank: 99,
            estimated_value: 100.0,
            uncertainty: 0.0,
            feasible: false,
        },
    ]
}

fn main() {
    let evidence = TransitionEvidence::from_selection(
        20260810,
        7,
        30,
        0.10,
        ExplorationReason::RankWeighted,
        &fixture(),
    )
    .and_then(|evidence| {
        evidence.validate_for_learning()?;
        Ok(evidence)
    });

    match evidence {
        Ok(evidence) => println!("{}", evidence.to_json()),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persists_probability_value_uncertainty_reason_and_feasibility() {
        let evidence = TransitionEvidence::from_selection(
            99,
            1234,
            30,
            0.10,
            ExplorationReason::Uncertainty,
            &fixture(),
        )
        .unwrap();
        evidence.validate_for_learning().unwrap();
        assert_eq!(evidence.selected_rank, 2);
        assert_eq!(evidence.estimated_value, 3.0);
        assert_eq!(evidence.uncertainty, 2.0);
        assert_eq!(evidence.behavior_probability, 0.10);
        assert_eq!(evidence.feasible_action_ids, vec![10, 20, 30]);
        assert!(evidence.to_json().contains("\"explorationReason\":\"uncertainty\""));
    }

    #[test]
    fn rejects_fabricated_log_probability() {
        let mut evidence = TransitionEvidence::from_selection(
            1,
            2,
            20,
            0.20,
            ExplorationReason::RankWeighted,
            &fixture(),
        )
        .unwrap();
        evidence.behavior_log_probability = -999.0;
        assert!(evidence.validate_for_learning().is_err());
    }

    #[test]
    fn rejects_infeasible_selected_action() {
        let result = TransitionEvidence::from_selection(
            1,
            2,
            40,
            0.01,
            ExplorationReason::Epsilon,
            &fixture(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn rejects_probability_that_cannot_be_an_sampling_likelihood() {
        for invalid in [0.0, -0.1, 1.1, f64::NAN] {
            assert!(TransitionEvidence::from_selection(
                1,
                2,
                10,
                invalid,
                ExplorationReason::Boltzmann,
                &fixture(),
            )
            .is_err());
        }
    }

    #[test]
    fn production_greedy_transition_is_explicit() {
        let evidence = TransitionEvidence::from_selection(
            5,
            6,
            10,
            1.0,
            ExplorationReason::Greedy,
            &fixture(),
        )
        .unwrap();
        evidence.validate_for_learning().unwrap();
        assert_eq!(evidence.behavior_probability, 1.0);
        assert_eq!(evidence.exploration_reason, ExplorationReason::Greedy);
    }
}
