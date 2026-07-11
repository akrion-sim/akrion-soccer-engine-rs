//! Fit field-conditioned reward utility heads from factual event/future pairs.
//!
//! A hermetic state-value head first learns terminal match outcome from canonical
//! 256-d all-players-plus-ball embeddings. Each typed reward utility then learns
//! the value change from its exact event state to a later field state. Authored
//! reward magnitudes are never training labels. Heads are fitted outside the
//! policy learner and frozen as a schema-v1 `DD_SOCCER_REWARD_CONTEXT_PATH`
//! artifact.

use serde::{Deserialize, Serialize};
use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMatch, SoccerRewardContextSample,
    SOCCER_MOMENT_EMBEDDING_DIM,
};
use std::collections::HashMap;

const MIN_UTILITY_SCALE: f64 = 0.0001;
const MAX_UTILITY_SCALE: f64 = 4.0;
const VALUE_DELTA_SCALE_FLOOR: f64 = 0.05;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RewardContextHead {
    bias: f64,
    weights: Vec<f64>,
}

const VALUE_HIDDEN_DIM: usize = 32;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct NeuralStateValueHead {
    hidden_biases: Vec<f64>,
    hidden_weights: Vec<Vec<f64>>,
    future_projection: Vec<Vec<f64>>,
    output_bias: f64,
    output_weights: Vec<f64>,
}

impl NeuralStateValueHead {
    fn deterministic(seed: u64) -> Self {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((state >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0) * 0.08
        };
        let hidden_weights: Vec<Vec<f64>> = (0..VALUE_HIDDEN_DIM)
            .map(|_| (0..SOCCER_MOMENT_EMBEDDING_DIM).map(|_| next()).collect())
            .collect();
        let future_projection = hidden_weights.clone();
        Self {
            hidden_biases: vec![0.0; VALUE_HIDDEN_DIM],
            hidden_weights,
            future_projection,
            output_bias: 0.0,
            output_weights: (0..VALUE_HIDDEN_DIM).map(|_| next()).collect(),
        }
    }

    fn valid(&self) -> bool {
        self.hidden_biases.len() == VALUE_HIDDEN_DIM
            && self.hidden_weights.len() == VALUE_HIDDEN_DIM
            && self
                .hidden_weights
                .iter()
                .all(|row| row.len() == SOCCER_MOMENT_EMBEDDING_DIM)
            && self.future_projection.len() == VALUE_HIDDEN_DIM
            && self
                .future_projection
                .iter()
                .all(|row| row.len() == SOCCER_MOMENT_EMBEDDING_DIM)
            && self.output_weights.len() == VALUE_HIDDEN_DIM
    }

    fn hidden(&self, embedding: &[f64]) -> Vec<f64> {
        if !self.valid() || embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM {
            return vec![0.0; VALUE_HIDDEN_DIM];
        }
        let norm = (SOCCER_MOMENT_EMBEDDING_DIM as f64).sqrt();
        self.hidden_weights
            .iter()
            .zip(&self.hidden_biases)
            .map(|(weights, bias)| {
                (weights
                    .iter()
                    .zip(embedding)
                    .fold(*bias, |sum, (weight, value)| sum + weight * value)
                    / norm)
                    .tanh()
            })
            .collect()
    }

    fn predict(&self, embedding: &[f64]) -> f64 {
        let hidden = self.hidden(embedding);
        let norm = (VALUE_HIDDEN_DIM as f64).sqrt();
        (self
            .output_weights
            .iter()
            .zip(hidden)
            .fold(self.output_bias, |sum, (weight, value)| {
                sum + weight * value
            })
            / norm)
            .tanh()
    }

    fn pretrain_future(&mut self, embedding: &[f64], future: &[f64], learning_rate: f64) {
        if !self.valid()
            || embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM
            || future.len() != SOCCER_MOMENT_EMBEDDING_DIM
        {
            return;
        }
        let hidden = self.hidden(embedding);
        let norm = (SOCCER_MOMENT_EMBEDDING_DIM as f64).sqrt();
        for hidden_index in 0..VALUE_HIDDEN_DIM {
            let target = (self.future_projection[hidden_index]
                .iter()
                .zip(future)
                .fold(0.0, |sum, (weight, value)| sum + weight * value)
                / norm)
                .tanh();
            let hidden_value = hidden[hidden_index];
            let delta =
                ((hidden_value - target) * (1.0 - hidden_value * hidden_value)).clamp(-2.0, 2.0);
            self.hidden_biases[hidden_index] =
                (self.hidden_biases[hidden_index] - learning_rate * delta / norm).clamp(-2.0, 2.0);
            for (weight, value) in self.hidden_weights[hidden_index].iter_mut().zip(embedding) {
                *weight = (*weight - learning_rate * delta * value / norm).clamp(-2.0, 2.0);
            }
        }
    }

    fn train_value_output(&mut self, embedding: &[f64], target: f64, learning_rate: f64) {
        if !self.valid() || embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM {
            return;
        }
        let hidden = self.hidden(embedding);
        let prediction = self.predict(embedding);
        let output_delta =
            ((prediction - target) * (1.0 - prediction * prediction)).clamp(-2.0, 2.0);
        let hidden_norm = (VALUE_HIDDEN_DIM as f64).sqrt();
        self.output_bias =
            (self.output_bias - learning_rate * output_delta / hidden_norm).clamp(-2.0, 2.0);
        for (weight, hidden_value) in self.output_weights.iter_mut().zip(hidden) {
            *weight = (*weight - learning_rate * output_delta * hidden_value / hidden_norm)
                .clamp(-2.0, 2.0);
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RewardContextArtifact {
    schema_version: u32,
    embedding_dim: usize,
    min_scale: f64,
    max_scale: f64,
    games: usize,
    minutes: f64,
    seed_base: u32,
    samples_by_kind: HashMap<String, usize>,
    #[serde(default)]
    effective_samples_by_kind: HashMap<String, f64>,
    #[serde(default)]
    value_delta_rms_by_kind: HashMap<String, f64>,
    #[serde(default)]
    state_value_head: RewardContextHead,
    #[serde(default)]
    neural_state_value_head: NeuralStateValueHead,
    #[serde(default)]
    state_value_reliability: f64,
    by_kind: HashMap<String, RewardContextHead>,
}

#[derive(Clone, Copy)]
struct KindSpec {
    name: &'static str,
    outcome_sign: f64,
}

const KIND_SPECS: &[KindSpec] = &[
    KindSpec {
        name: "ShotAttempt",
        outcome_sign: 1.0,
    },
    KindSpec {
        name: "ShotOnTarget",
        outcome_sign: 1.0,
    },
    KindSpec {
        name: "ShotOffTargetPenalty",
        outcome_sign: -1.0,
    },
    KindSpec {
        name: "CompletedForwardPass",
        outcome_sign: 1.0,
    },
    KindSpec {
        name: "BadPassChainPenalty",
        outcome_sign: -1.0,
    },
    KindSpec {
        name: "TurnoverChainBlame",
        outcome_sign: -1.0,
    },
    KindSpec {
        name: "ProgressiveCarryContinuation",
        outcome_sign: 1.0,
    },
    KindSpec {
        name: "OverdribbleDispossession",
        outcome_sign: -1.0,
    },
    KindSpec {
        name: "DefensiveDispossession",
        outcome_sign: 1.0,
    },
];

struct LabeledMoment {
    game_index: usize,
    sample: SoccerRewardContextSample,
    outcome: f64,
    value_delta: Option<f64>,
    /// Inverse occurrence count for this `(match, team, event kind)` group.
    /// Each team-match therefore contributes at most one unit of evidence to a
    /// head, rather than repeated carries dominating rare decisive events.
    sample_weight: f64,
    /// Inverse number of factual events for this team-match. This prevents the
    /// state-value learner from treating a high-event match as many independent
    /// terminal outcomes.
    value_weight: f64,
}

fn predict_head(head: &RewardContextHead, embedding: &[f64]) -> f64 {
    if embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM {
        return 0.0;
    }
    let norm = (SOCCER_MOMENT_EMBEDDING_DIM as f64).sqrt();
    head.weights
        .iter()
        .zip(embedding)
        .fold(head.bias, |sum, (weight, value)| sum + weight * value)
        / norm
}

fn fit_state_value_head(
    rows: &[&LabeledMoment],
    prior: Option<&RewardContextHead>,
    epochs: usize,
    learning_rate: f64,
) -> RewardContextHead {
    let mut head = prior
        .filter(|head| head.weights.len() == SOCCER_MOMENT_EMBEDDING_DIM)
        .cloned()
        .unwrap_or_else(|| RewardContextHead {
            bias: 0.0,
            weights: vec![0.0; SOCCER_MOMENT_EMBEDDING_DIM],
        });
    let norm = (SOCCER_MOMENT_EMBEDDING_DIM as f64).sqrt();
    let ridge = 1e-4;
    for _ in 0..epochs.max(1) {
        for row in rows {
            let prediction = predict_head(&head, &row.sample.embedding);
            let error = (prediction - row.outcome).clamp(-2.0, 2.0);
            let weighted_rate = learning_rate * row.value_weight;
            head.bias = (head.bias - weighted_rate * error / norm).clamp(-2.0, 2.0);
            for (weight, value) in head.weights.iter_mut().zip(&row.sample.embedding) {
                if value.is_finite() {
                    *weight = (*weight - weighted_rate * (error * value / norm + ridge * *weight))
                        .clamp(-2.0, 2.0);
                }
            }
        }
    }
    head
}

fn fit_neural_state_value_head(
    rows: &[&LabeledMoment],
    prior: Option<&NeuralStateValueHead>,
    epochs: usize,
    learning_rate: f64,
    seed: u64,
) -> NeuralStateValueHead {
    let mut head = prior
        .filter(|head| head.valid())
        .cloned()
        .unwrap_or_else(|| NeuralStateValueHead::deterministic(seed));
    for _ in 0..epochs.max(1) {
        for row in rows {
            if let Some(future) = row.sample.future_embedding.as_deref() {
                head.pretrain_future(
                    &row.sample.embedding,
                    future,
                    learning_rate * row.sample_weight,
                );
            }
        }
    }
    for _ in 0..epochs.max(1) {
        for row in rows {
            head.train_value_output(
                &row.sample.embedding,
                row.outcome,
                learning_rate * row.value_weight,
            );
        }
    }
    head
}

fn fit_head(
    rows: &[&LabeledMoment],
    outcome_sign: f64,
    delta_rms: f64,
    prior: Option<&RewardContextHead>,
    epochs: usize,
    learning_rate: f64,
) -> RewardContextHead {
    let mut head = prior
        .filter(|head| head.weights.len() == SOCCER_MOMENT_EMBEDDING_DIM)
        .cloned()
        .unwrap_or_else(|| RewardContextHead {
            bias: 0.0,
            weights: vec![0.0; SOCCER_MOMENT_EMBEDDING_DIM],
        });
    if rows.is_empty() {
        return head;
    }
    let norm = (SOCCER_MOMENT_EMBEDDING_DIM as f64).sqrt();
    let ridge = 1e-4;
    for _ in 0..epochs.max(1) {
        for row in rows {
            if row.sample.embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM {
                continue;
            }
            // Hermetic TD target: utility follows the learned whole-field value
            // change after the factual event, never its hand-authored magnitude.
            let Some(value_delta) = row.value_delta else {
                continue;
            };
            let target = (outcome_sign * value_delta / delta_rms.max(1e-6)).clamp(-1.0, 1.0);
            let prediction = predict_head(&head, &row.sample.embedding);
            let error = (prediction - target).clamp(-2.0, 2.0);
            let weighted_rate = learning_rate * row.sample_weight;
            head.bias = (head.bias - weighted_rate * error / norm).clamp(-2.0, 2.0);
            for (weight, value) in head.weights.iter_mut().zip(&row.sample.embedding) {
                if value.is_finite() {
                    *weight = (*weight - weighted_rate * (error * value / norm + ridge * *weight))
                        .clamp(-2.0, 2.0);
                }
            }
        }
    }
    head
}

fn parse_hex(raw: Option<&String>, default: u32) -> u32 {
    raw.and_then(|value| u32::from_str_radix(value.trim_start_matches("0x"), 16).ok())
        .unwrap_or(default)
}

fn main() {
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect(
        "usage: soccer_reward_context_fit <out.json> [games=40] [minutes=1] [seed_hex] [prior.json]",
    );
    let games = args
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(40usize);
    let minutes = args
        .get(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0f64);
    let seed_base = parse_hex(args.get(4), 0xC07E_0000);
    let prior: Option<RewardContextArtifact> = args.get(5).and_then(|path| {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
    });
    let epochs = std::env::var("SOCCER_REWARD_CONTEXT_EPOCHS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6usize);
    let learning_rate = std::env::var("SOCCER_REWARD_CONTEXT_LEARNING_RATE")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &f64| value.is_finite() && *value > 0.0)
        .unwrap_or(0.02);

    let mut moments = Vec::new();
    for game in 0..games {
        let mut config = MatchConfig {
            duration_seconds: minutes * 60.0,
            learning_enabled: true,
            full_game_learning_enabled: true,
            seed: seed_base.wrapping_add(game as u32),
            ..MatchConfig::default()
        };
        config.retrieval.capture_enabled = true;
        config.retrieval.outcome_horizon = 45;
        let mut sim = SoccerMatch::default_11v11(config);
        sim.set_uniform_elite_players();
        for _ in 0..sim.config.total_ticks() {
            sim.run_time_step();
        }
        let summary = sim.summary();
        let home_margin = summary.score_home as i32 - summary.score_away as i32;
        let before = moments.len();
        let valid_samples: Vec<_> = sim
            .reward_context_samples()
            .iter()
            .filter(|sample| {
                sample.embedding.len() == SOCCER_MOMENT_EMBEDDING_DIM
                    && sample.embedding.iter().all(|value| value.is_finite())
            })
            .cloned()
            .collect();
        let mut occurrence_counts = HashMap::<(String, bool), usize>::new();
        let mut team_event_counts = HashMap::<bool, usize>::new();
        for sample in &valid_samples {
            let home = matches!(sample.team, soccer_engine::des::general::soccer::Team::Home);
            *occurrence_counts
                .entry((sample.kind.clone(), home))
                .or_default() += 1;
            *team_event_counts.entry(home).or_default() += 1;
        }
        moments.extend(valid_samples.into_iter().filter_map(|sample| {
            let home = matches!(sample.team, soccer_engine::des::general::soccer::Team::Home);
            let occurrence_count = *occurrence_counts
                .get(&(sample.kind.clone(), home))
                .unwrap_or(&1);
            let team_event_count = *team_event_counts.get(&home).unwrap_or(&1);
            if sample.embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM
                || sample.embedding.iter().any(|value| !value.is_finite())
            {
                return None;
            }
            let margin = match sample.team {
                soccer_engine::des::general::soccer::Team::Home => home_margin,
                soccer_engine::des::general::soccer::Team::Away => -home_margin,
            };
            let result = margin.signum() as f64;
            let margin_bonus = (margin as f64 / 3.0).clamp(-1.0, 1.0) * 0.20;
            Some(LabeledMoment {
                game_index: game,
                sample,
                outcome: (result + margin_bonus).clamp(-1.0, 1.0),
                value_delta: None,
                sample_weight: 1.0 / occurrence_count as f64,
                value_weight: 1.0 / team_event_count as f64,
            })
        }));
        eprintln!(
            "context_fit game {}/{} moments_added={} total={}",
            game + 1,
            games,
            moments.len() - before,
            moments.len()
        );
    }

    let all_rows: Vec<_> = moments.iter().collect();
    let even_rows: Vec<_> = moments
        .iter()
        .filter(|row| row.game_index % 2 == 0)
        .collect();
    let odd_rows: Vec<_> = moments
        .iter()
        .filter(|row| row.game_index % 2 == 1)
        .collect();
    let value_seed = seed_base as u64 ^ 0x5A17_EF1E_1D00_0001;
    let prior_neural_value = prior
        .as_ref()
        .map(|artifact| &artifact.neural_state_value_head)
        .filter(|head| head.valid());
    let value_for_even = fit_neural_state_value_head(
        &odd_rows,
        prior_neural_value,
        epochs,
        learning_rate,
        value_seed,
    );
    let value_for_odd = fit_neural_state_value_head(
        &even_rows,
        prior_neural_value,
        epochs,
        learning_rate,
        value_seed,
    );
    let mut baseline_error = 0.0;
    let mut model_error = 0.0;
    let mut validation_weight = 0.0;
    let cross_deltas: Vec<_> = moments
        .iter()
        .map(|row| {
            let head = if row.game_index % 2 == 0 {
                &value_for_even
            } else {
                &value_for_odd
            };
            let prediction = head.predict(&row.sample.embedding);
            baseline_error += row.value_weight * row.outcome.powi(2);
            model_error += row.value_weight * (prediction - row.outcome).powi(2);
            validation_weight += row.value_weight;
            row.sample
                .future_embedding
                .as_deref()
                .map(|future| head.predict(future) - head.predict(&row.sample.embedding))
        })
        .collect();
    let state_value_reliability = if validation_weight > 0.0 && baseline_error > 1e-9 {
        ((baseline_error - model_error) / baseline_error).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let state_value_head = fit_state_value_head(
        &all_rows,
        prior.as_ref().map(|artifact| &artifact.state_value_head),
        epochs,
        learning_rate,
    );
    let neural_state_value_head = fit_neural_state_value_head(
        &all_rows,
        prior
            .as_ref()
            .map(|artifact| &artifact.neural_state_value_head),
        epochs,
        learning_rate,
        value_seed,
    );
    for (row, delta) in moments.iter_mut().zip(cross_deltas) {
        row.value_delta = delta
            .map(|value| value * state_value_reliability)
            .filter(|value| value.is_finite());
    }

    let mut by_kind = HashMap::new();
    let mut samples_by_kind = HashMap::new();
    let mut effective_samples_by_kind = HashMap::new();
    let mut value_delta_rms_by_kind = HashMap::new();
    for spec in KIND_SPECS {
        let rows: Vec<_> = moments
            .iter()
            .filter(|row| row.sample.kind == spec.name && row.value_delta.is_some())
            .collect();
        samples_by_kind.insert(spec.name.to_string(), rows.len());
        effective_samples_by_kind.insert(
            spec.name.to_string(),
            rows.iter().map(|row| row.sample_weight).sum(),
        );
        let weight_sum = rows.iter().map(|row| row.sample_weight).sum::<f64>();
        let delta_rms = if weight_sum > 0.0 {
            (rows
                .iter()
                .map(|row| row.sample_weight * row.value_delta.unwrap_or(0.0).powi(2))
                .sum::<f64>()
                / weight_sum)
                .sqrt()
        } else {
            1.0
        };
        value_delta_rms_by_kind.insert(spec.name.to_string(), delta_rms);
        let prior_head = prior
            .as_ref()
            .and_then(|artifact| artifact.by_kind.get(spec.name));
        by_kind.insert(
            spec.name.to_string(),
            fit_head(
                &rows,
                spec.outcome_sign,
                delta_rms.max(VALUE_DELTA_SCALE_FLOOR),
                prior_head,
                epochs,
                learning_rate,
            ),
        );
    }

    let artifact = RewardContextArtifact {
        schema_version: 1,
        embedding_dim: SOCCER_MOMENT_EMBEDDING_DIM,
        min_scale: MIN_UTILITY_SCALE,
        max_scale: MAX_UTILITY_SCALE,
        games,
        minutes,
        seed_base,
        samples_by_kind,
        effective_samples_by_kind,
        value_delta_rms_by_kind,
        state_value_head,
        neural_state_value_head,
        state_value_reliability,
        by_kind,
    };
    let json = serde_json::to_string_pretty(&artifact).expect("serialize context artifact");
    std::fs::write(out, format!("{json}\n")).expect("write context artifact");
    println!(
        "wrote {out} moments={} heads={} baseline=1 clamp=[{},{}]",
        moments.len(),
        artifact.by_kind.len(),
        MIN_UTILITY_SCALE,
        MAX_UTILITY_SCALE
    );
}
