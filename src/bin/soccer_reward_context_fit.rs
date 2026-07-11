//! Fit field-conditioned reward utility heads from realised configuration moments.
//!
//! The data source is the existing whole-field moment corpus: every row contains
//! the canonical 256-d all-players-plus-ball embedding, the decision, and its
//! realised n-step return. Heads are fitted outside the policy learner and then
//! frozen as a schema-v1 `DD_SOCCER_REWARD_CONTEXT_PATH` artifact.

use serde::{Deserialize, Serialize};
use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerConfigMomentInsert, SoccerMatch,
    SOCCER_MOMENT_EMBEDDING_DIM,
};
use std::collections::HashMap;

const MIN_UTILITY_SCALE: f64 = 0.0001;
const MAX_UTILITY_SCALE: f64 = 4.0;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RewardContextHead {
    bias: f64,
    weights: Vec<f64>,
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
    by_kind: HashMap<String, RewardContextHead>,
}

#[derive(Clone, Copy)]
struct KindSpec {
    name: &'static str,
    family: &'static str,
    outcome_sign: f64,
}

const KIND_SPECS: &[KindSpec] = &[
    KindSpec { name: "ShotAttempt", family: "shot", outcome_sign: 1.0 },
    KindSpec { name: "ShotOnTarget", family: "shot", outcome_sign: 1.0 },
    KindSpec { name: "ShotOffTargetPenalty", family: "shot", outcome_sign: -1.0 },
    KindSpec { name: "CompletedForwardPass", family: "pass", outcome_sign: 1.0 },
    KindSpec { name: "BadPassChainPenalty", family: "pass", outcome_sign: -1.0 },
    KindSpec { name: "TurnoverChainBlame", family: "pass", outcome_sign: -1.0 },
    KindSpec { name: "ProgressiveCarryContinuation", family: "dribble", outcome_sign: 1.0 },
    KindSpec { name: "OverdribbleDispossession", family: "dribble", outcome_sign: -1.0 },
    KindSpec { name: "DefensiveDispossession", family: "defend", outcome_sign: 1.0 },
];

fn normalized_action_family(action: &str) -> &'static str {
    let action = action.to_ascii_lowercase();
    if action.contains("shoot") || action.contains("header") {
        "shot"
    } else if action.contains("pass") || action.contains("cross") || action.contains("through") {
        "pass"
    } else if action.contains("dribble") || action.contains("carry") || action.contains("nutmeg") {
        "dribble"
    } else if action.contains("defend") || action.contains("tackle") || action.contains("press") {
        "defend"
    } else {
        "other"
    }
}

fn fit_head(
    rows: &[&SoccerConfigMomentInsert],
    outcome_sign: f64,
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
            if row.embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM {
                continue;
            }
            // Bounded outcome target: good future for a reward raises utility;
            // bad future for a penalty raises its magnitude (outcome_sign=-1).
            let target = (outcome_sign * row.nstep_return / 20.0).tanh();
            let prediction = head
                .weights
                .iter()
                .zip(&row.embedding)
                .fold(head.bias, |sum, (weight, value)| sum + weight * value)
                / norm;
            let error = (prediction - target).clamp(-2.0, 2.0);
            head.bias = (head.bias - learning_rate * error / norm).clamp(-2.0, 2.0);
            for (weight, value) in head.weights.iter_mut().zip(&row.embedding) {
                if value.is_finite() {
                    *weight = (*weight - learning_rate * (error * value / norm + ridge * *weight))
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
    let games = args.get(2).and_then(|value| value.parse().ok()).unwrap_or(40usize);
    let minutes = args.get(3).and_then(|value| value.parse().ok()).unwrap_or(1.0f64);
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
        let before = moments.len();
        moments.extend(sim.config_moments().into_iter().filter(|row| {
            row.embedding.len() == SOCCER_MOMENT_EMBEDDING_DIM
                && row.embedding.iter().all(|value| value.is_finite())
                && row.nstep_return.is_finite()
        }));
        eprintln!(
            "context_fit game {}/{} moments_added={} total={}",
            game + 1,
            games,
            moments.len() - before,
            moments.len()
        );
    }

    let mut by_kind = HashMap::new();
    let mut samples_by_kind = HashMap::new();
    for spec in KIND_SPECS {
        let rows: Vec<_> = moments
            .iter()
            .filter(|row| normalized_action_family(&row.action) == spec.family)
            .collect();
        samples_by_kind.insert(spec.name.to_string(), rows.len());
        let prior_head = prior.as_ref().and_then(|artifact| artifact.by_kind.get(spec.name));
        by_kind.insert(
            spec.name.to_string(),
            fit_head(&rows, spec.outcome_sign, prior_head, epochs, learning_rate),
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
