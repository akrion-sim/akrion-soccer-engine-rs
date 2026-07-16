//! Monotonic-climb RATCHET harness — the structural answer to "make sure we actually climb, no
//! stalling/plateaus/divergence".
//!
//! Trains the neural policy in INCREMENTS (analytic-opponent carry-forward by default), and after
//! every increment EVALS the current checkpoint head-to-head vs the pure-ANALYTIC engine
//! (candidate net on one side, no net / analytic on the other, sides alternated for fairness). It
//! KEEPS THE BEST checkpoint and discards regressions — so the kept policy is monotonic BY
//! CONSTRUCTION: it can plateau, but it can never go backwards, no matter how noisy or divergent a
//! given training window is. The printed climb curve (eval_mean per increment) reveals a stall or
//! divergence the instant it happens instead of only at the end.
//!
//! Run:
//!   cargo run --release --bin soccer_climb_ratchet -- [increment_games] [minutes] [eval_games] [increments] [seed_base]
//!   # anti-parity defaults are enabled unless overridden:
//!   # neural-authoritative, MC critic target, target standardization, analytic opponents.
//!   cargo run --release --bin soccer_climb_ratchet -- 5 2 10 8 0x5EED0000

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::time::Duration;

use soccer_engine::des::general::soccer::{
    average_soccer_neural_network_snapshots, enable_deterministic_formation_lp, MatchConfig,
    SoccerMarlAlgorithm, SoccerMatch, SoccerNeuralBlendMode, SoccerNeuralLearningBackend,
    SoccerNeuralLearningConfig, SoccerNeuralNetworkSnapshot, SoccerQPolicyOptions,
    SoccerTeamQPolicies, Team,
};
use soccer_engine::des::general::soccer_eval_gate::wilson_lower_bound;

fn env_bool(name: &str, default: bool) -> bool {
    let Ok(raw) = std::env::var(name) else {
        return default;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => {
            eprintln!("warning: invalid {name}={raw:?}; using default {default}");
            default
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    let Ok(raw) = std::env::var(name) else {
        return default;
    };
    raw.trim().parse().unwrap_or_else(|_| {
        eprintln!("warning: invalid {name}={raw:?}; using default {default}");
        default
    })
}

fn env_usize_any(names: &[&str], default: usize) -> usize {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().map(|raw| (name, raw)))
        .map(|(name, raw)| {
            raw.trim().parse().unwrap_or_else(|_| {
                eprintln!("warning: invalid {name}={raw:?}; using default {default}");
                default
            })
        })
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    let Ok(raw) = std::env::var(name) else {
        return default;
    };
    match raw.trim().parse::<f64>() {
        Ok(value) if value.is_finite() => value,
        _ => {
            eprintln!("warning: invalid {name}={raw:?}; using default {default}");
            default
        }
    }
}

fn env_f64_any(names: &[&str], default: f64) -> f64 {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().map(|raw| (name, raw)))
        .map(|(name, raw)| match raw.trim().parse::<f64>() {
            Ok(value) if value.is_finite() => value,
            _ => {
                eprintln!("warning: invalid {name}={raw:?}; using default {default}");
                default
            }
        })
        .unwrap_or(default)
}

fn env_f64_list(name: &str) -> Vec<f64> {
    std::env::var(name)
        .ok()
        .into_iter()
        .flat_map(|raw| {
            raw.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
                .filter_map(|part| part.trim().parse::<f64>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .collect::<Vec<_>>()
        })
        .take(8)
        .collect()
}

fn climb_learning_rate_candidates() -> Vec<f64> {
    let mut values = env_f64_list("SOCCER_CLIMB_LEARNING_RATE_CANDIDATES");
    if values.is_empty() {
        if let Some(value) = std::env::var("SOCCER_NEURAL_LEARNING_RATE")
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            values.push(value);
        }
    }
    if values.is_empty() {
        values.extend([0.015, 0.025]);
    }
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);
    values
}

fn climb_authoritative_lambda_candidates(default_lambda: f64) -> Vec<f64> {
    let mut values = env_f64_list("SOCCER_CLIMB_AUTHORITATIVE_LAMBDA_CANDIDATES");
    if values.is_empty() {
        values.push(default_lambda.max(0.5));
    }
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-12);
    values
}

fn climb_uniform_elite_players_enabled() -> bool {
    env_bool("SOCCER_ENGINE_UNIFORM_ELITE_PLAYERS", true)
}

fn env_default_bool(name: &str, default: bool) -> bool {
    let value = env_bool(name, default);
    if std::env::var(name).is_err() {
        std::env::set_var(name, if value { "1" } else { "0" });
    }
    value
}

fn env_default_f64(name: &str, default: f64) -> f64 {
    let value = env_f64(name, default);
    if std::env::var(name).is_err() {
        std::env::set_var(name, value.to_string());
    }
    value
}

fn env_default_usize(name: &str, default: usize) -> usize {
    let value = env_usize(name, default);
    if std::env::var(name).is_err() {
        std::env::set_var(name, value.to_string());
    }
    value
}

fn parse_seed_arg(raw: Option<&String>, default: u32) -> u32 {
    let Some(raw) = raw else {
        return default;
    };
    let trimmed = raw.trim();
    let hex = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"));
    if let Some(hex) = hex {
        u32::from_str_radix(hex, 16).unwrap_or(default)
    } else if trimmed
        .chars()
        .any(|ch| ch.is_ascii_hexdigit() && ch.is_ascii_alphabetic())
    {
        u32::from_str_radix(trimmed, 16).unwrap_or(default)
    } else {
        trimmed.parse().unwrap_or(default)
    }
}

fn parse_seed_value(raw: &str, default: u32) -> u32 {
    parse_seed_arg(Some(&raw.to_string()), default)
}

fn climb_seed_bases(default_seed: u32) -> Vec<u32> {
    let mut seeds = std::env::var("SOCCER_CLIMB_SEED_BASES")
        .ok()
        .into_iter()
        .flat_map(|raw| {
            raw.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
                .filter(|part| !part.trim().is_empty())
                .map(|part| parse_seed_value(part.trim(), default_seed))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if seeds.is_empty() {
        seeds.push(default_seed);
    }
    seeds
}

fn climb_neural_config(learning_rate: f64) -> SoccerNeuralLearningConfig {
    let mut neural = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Inline,
        marl_algorithm: SoccerMarlAlgorithm::Mappo,
        ..SoccerNeuralLearningConfig::default()
    };
    neural.learning_rate = learning_rate;
    neural.optimizer_momentum = env_f64(
        "SOCCER_NEURAL_OPTIMIZER_MOMENTUM",
        neural.optimizer_momentum,
    );
    neural.mappo_team_reward_share = env_f64("SOCCER_MAPPO_TEAM_REWARD_SHARE", 0.50);
    neural.marl_team_reward_weight = env_f64(
        "SOCCER_MARL_TEAM_REWARD_WEIGHT",
        neural.marl_team_reward_weight,
    );
    neural.marl_intermediate_reward_weight = env_f64(
        "SOCCER_MARL_INTERMEDIATE_REWARD_WEIGHT",
        neural.marl_intermediate_reward_weight,
    );
    neural.mappo_clip_epsilon = env_f64("SOCCER_MAPPO_CLIP_EPSILON", neural.mappo_clip_epsilon);
    neural.train_every_ticks = env_usize("SOCCER_NEURAL_TRAIN_EVERY_TICKS", 1).max(1);
    neural.max_batches_per_tick = env_usize("SOCCER_NEURAL_MAX_BATCHES_PER_TICK", 4).max(1);
    neural.replay_capacity = env_usize("SOCCER_NEURAL_REPLAY_CAPACITY", 8192).max(1);
    neural.replay_samples_per_tick = env_usize("SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK", 128).max(1);
    neural.batch_size = env_usize("SOCCER_NEURAL_BATCH_SIZE", 64).max(1);
    neural.hidden_units = env_usize("SOCCER_NEURAL_HIDDEN_UNITS", 96).max(1);
    neural.target_scale = env_f64("SOCCER_NEURAL_TARGET_SCALE", 120.0);
    neural.target_clip = env_f64("SOCCER_NEURAL_TARGET_CLIP", neural.target_clip);
    neural.target_popart_enabled = env_bool("SOCCER_NEURAL_TARGET_POPART", true);
    neural.lp_coupling_enabled = env_bool("SOCCER_NEURAL_LP_COUPLING_ENABLED", true);
    neural
}

fn configure_authoritative_blend(config: &mut MatchConfig, authoritative_lambda: f64) {
    config.neural_blend.mode = SoccerNeuralBlendMode::Authoritative;
    config.neural_blend.actor_critic = true;
    config.neural_blend.lambda =
        env_f64("SOCCER_NEURAL_BLEND_LAMBDA", authoritative_lambda.max(0.5));
    config.neural_blend.warmup_steps = env_usize("SOCCER_NEURAL_BLEND_WARMUP_STEPS", 1).max(1);
    config.neural_blend.candidates = env_usize("SOCCER_NEURAL_BLEND_CANDIDATES", 8).clamp(2, 16);
    config.neural_blend.mcts_enabled = env_bool("SOCCER_CLIMB_NEURAL_MCTS_ENABLED", false)
        && std::env::var("SOCCER_DISABLE_NEURAL_MCTS").ok().as_deref() != Some("1");
    config.neural_blend.mcts_simulations = env_usize_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_SIMULATIONS",
            "SOCCER_MCTS_SIMULATIONS",
        ],
        8,
    )
    .clamp(1, 32);
    config.neural_blend.mcts_candidates = env_usize_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_CANDIDATES",
            "SOCCER_MCTS_CANDIDATES",
        ],
        4,
    )
    .clamp(2, 16);
    config.neural_blend.mcts_depth =
        env_usize_any(&["SOCCER_CLIMB_NEURAL_MCTS_DEPTH", "SOCCER_MCTS_DEPTH"], 1).clamp(1, 3);
    config.neural_blend.mcts_exploration = env_f64_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_EXPLORATION",
            "SOCCER_MCTS_EXPLORATION",
        ],
        1.25,
    )
    .clamp(0.0, 4.0);
    config.neural_blend.mcts_model_weight = env_f64_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_MODEL_WEIGHT",
            "SOCCER_MCTS_MODEL_WEIGHT",
        ],
        0.35,
    )
    .clamp(0.0, 1.0);

    // The climb harness must evaluate the same POMDP -> neural -> QP/MPC execution
    // stack that production serves. Previously the ratchet silently used the
    // MatchConfig defaults, which leave tier-2 MPC off; a policy could therefore
    // look better in the harness while bypassing the controller it must drive live.
    let mpc_enabled = env_bool("SOCCER_CLIMB_MPC_ENABLED", true);
    config.formation_lp_enabled = env_bool("SOCCER_CLIMB_FORMATION_LP_ENABLED", true);
    config.mpc.tier2_player_enabled = mpc_enabled;
    config.mpc.field_aware_enabled =
        mpc_enabled && env_bool("SOCCER_CLIMB_MPC_FIELD_AWARE_ENABLED", true);
    config.mpc.reconcile_enabled =
        mpc_enabled && env_bool("SOCCER_CLIMB_MPC_RECONCILE_ENABLED", true);
    config.mpc.latent_objective_enabled =
        mpc_enabled && env_bool("SOCCER_CLIMB_MPC_LATENT_OBJECTIVE_ENABLED", true);
    config.mpc.player_horizon = env_usize("SOCCER_CLIMB_MPC_PLAYER_HORIZON", 30).max(1);
}

fn analytic_opponent_game_enabled(game_index: usize, fraction: f64) -> bool {
    let fraction = fraction.clamp(0.0, 1.0);
    if fraction <= 0.0 {
        return false;
    }
    if fraction >= 1.0 {
        return true;
    }
    let before = ((game_index as f64) * fraction).floor();
    let after = (((game_index + 1) as f64) * fraction).floor();
    after > before
}

fn climb_variant_seed(seed: u32, variant_index: usize, side_index: usize) -> u32 {
    seed.wrapping_add((variant_index as u32).wrapping_mul(0x9E37_79B9))
        .wrapping_add((side_index as u32) << 16)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SeedVariantMode {
    Average,
    Sequential,
}

impl SeedVariantMode {
    fn label(self) -> &'static str {
        match self {
            SeedVariantMode::Average => "average",
            SeedVariantMode::Sequential => "sequential",
        }
    }
}

fn climb_seed_variant_modes(default_include_sequential: bool) -> Vec<SeedVariantMode> {
    let mut modes = Vec::new();
    if let Ok(raw) = std::env::var("SOCCER_CLIMB_SEED_VARIANT_MODES") {
        for part in raw.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace()) {
            match part.trim().to_ascii_lowercase().as_str() {
                "average" | "avg" => modes.push(SeedVariantMode::Average),
                "sequential" | "seq" => modes.push(SeedVariantMode::Sequential),
                "both" => {
                    modes.push(SeedVariantMode::Average);
                    modes.push(SeedVariantMode::Sequential);
                }
                _ => {}
            }
        }
    } else if std::env::var("SOCCER_CLIMB_SEQUENTIAL_SEED_VARIANTS").is_ok() {
        modes.push(
            if env_bool("SOCCER_CLIMB_SEQUENTIAL_SEED_VARIANTS", false) {
                SeedVariantMode::Sequential
            } else {
                SeedVariantMode::Average
            },
        );
    }

    if modes.is_empty() {
        modes.push(SeedVariantMode::Average);
        if default_include_sequential {
            modes.push(SeedVariantMode::Sequential);
        }
    }
    modes.dedup();
    modes
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrainingSideMode {
    Paired,
    HomeOnly,
    AwayOnly,
    HomeAway,
    AwayHome,
}

impl TrainingSideMode {
    fn label(self) -> &'static str {
        match self {
            TrainingSideMode::Paired => "paired",
            TrainingSideMode::HomeOnly => "home",
            TrainingSideMode::AwayOnly => "away",
            TrainingSideMode::HomeAway => "home-away",
            TrainingSideMode::AwayHome => "away-home",
        }
    }
}

fn climb_training_side_modes(
    default_include_away: bool,
    paired_training_sides: bool,
    alternate_training_sides: bool,
) -> Vec<TrainingSideMode> {
    let mut modes = Vec::new();
    if let Ok(raw) = std::env::var("SOCCER_CLIMB_TRAIN_SIDE_MODES") {
        for part in raw.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace()) {
            match part.trim().to_ascii_lowercase().as_str() {
                "paired" | "pair" | "both-sides" => modes.push(TrainingSideMode::Paired),
                "home" | "home-only" => modes.push(TrainingSideMode::HomeOnly),
                "away" | "away-only" => modes.push(TrainingSideMode::AwayOnly),
                "home-away" | "homeaway" | "home-then-away" => {
                    modes.push(TrainingSideMode::HomeAway);
                }
                "away-home" | "awayhome" | "away-then-home" => {
                    modes.push(TrainingSideMode::AwayHome);
                }
                "all" => {
                    modes.push(TrainingSideMode::Paired);
                    modes.push(TrainingSideMode::HomeOnly);
                    modes.push(TrainingSideMode::AwayOnly);
                    modes.push(TrainingSideMode::HomeAway);
                    modes.push(TrainingSideMode::AwayHome);
                }
                _ => {}
            }
        }
    }

    if modes.is_empty() {
        if paired_training_sides {
            modes.push(TrainingSideMode::Paired);
            if default_include_away {
                modes.push(TrainingSideMode::AwayHome);
            }
        } else if alternate_training_sides {
            modes.push(TrainingSideMode::HomeOnly);
            modes.push(TrainingSideMode::AwayOnly);
        } else {
            modes.push(TrainingSideMode::HomeOnly);
        }
    }
    modes.dedup();
    modes
}

#[derive(Clone, Debug)]
struct RewardProfile {
    label: String,
    forward_pass: f64,
    pass_chain: f64,
    pass_turnover_penalty: f64,
    quick_forward_release: f64,
    shot_shaping: f64,
    shot_commitment: f64,
}

impl RewardProfile {
    fn scaled(
        &self,
        label: impl Into<String>,
        forward_pass: f64,
        pass_chain: f64,
        pass_turnover_penalty: f64,
        quick_forward_release: f64,
        shot_shaping: f64,
        shot_commitment_add: f64,
    ) -> Self {
        Self {
            label: label.into(),
            forward_pass: (self.forward_pass * forward_pass).clamp(0.0, 20.0),
            pass_chain: (self.pass_chain * pass_chain).clamp(0.0, 20.0),
            pass_turnover_penalty: (self.pass_turnover_penalty * pass_turnover_penalty)
                .clamp(1.0, 20.0),
            quick_forward_release: (self.quick_forward_release * quick_forward_release)
                .clamp(0.0, 2.0),
            shot_shaping: (self.shot_shaping * shot_shaping).clamp(0.0, 1.0),
            shot_commitment: (self.shot_commitment + shot_commitment_add).clamp(0.0, 10.0),
        }
    }

    fn apply(&self) {
        std::env::set_var(
            "DD_SOCCER_FORWARD_PASS_REWARD_SCALE",
            self.forward_pass.to_string(),
        );
        std::env::set_var(
            "DD_SOCCER_PASS_CHAIN_REWARD_SCALE",
            self.pass_chain.to_string(),
        );
        std::env::set_var(
            "DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE",
            self.pass_turnover_penalty.to_string(),
        );
        std::env::set_var(
            "DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE",
            self.quick_forward_release.to_string(),
        );
        std::env::set_var(
            "DD_SOCCER_SHOT_SHAPING_REWARD_SCALE",
            self.shot_shaping.to_string(),
        );
        std::env::set_var(
            "DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE",
            self.shot_commitment.to_string(),
        );
    }

    fn summary(&self) -> String {
        format!(
            "{}:fwd={:.2},chain={:.2},turn={:.2},quick={:.2},shot={:.2},commit={:.2}",
            self.label,
            self.forward_pass,
            self.pass_chain,
            self.pass_turnover_penalty,
            self.quick_forward_release,
            self.shot_shaping,
            self.shot_commitment
        )
    }
}

fn parse_reward_profile_spec(index: usize, spec: &str) -> Option<RewardProfile> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (label, values) = trimmed
        .split_once(':')
        .map(|(label, values)| (label.trim(), values))
        .unwrap_or(("profile", trimmed));
    let values = values
        .split(|ch: char| ch == ',' || ch == '/' || ch.is_whitespace())
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.len() != 6 {
        return None;
    }
    Some(RewardProfile {
        label: if label.is_empty() || label == "profile" {
            format!("profile{index}")
        } else {
            label.to_string()
        },
        forward_pass: values[0].clamp(0.0, 20.0),
        pass_chain: values[1].clamp(0.0, 20.0),
        pass_turnover_penalty: values[2].clamp(1.0, 20.0),
        quick_forward_release: values[3].clamp(0.0, 2.0),
        shot_shaping: values[4].clamp(0.0, 1.0),
        shot_commitment: values[5].clamp(0.0, 10.0),
    })
}

fn reward_profile_signature(profile: &RewardProfile) -> [i64; 6] {
    [
        (profile.forward_pass * 1_000_000.0).round() as i64,
        (profile.pass_chain * 1_000_000.0).round() as i64,
        (profile.pass_turnover_penalty * 1_000_000.0).round() as i64,
        (profile.quick_forward_release * 1_000_000.0).round() as i64,
        (profile.shot_shaping * 1_000_000.0).round() as i64,
        (profile.shot_commitment * 1_000_000.0).round() as i64,
    ]
}

fn push_unique_reward_profile(profiles: &mut Vec<RewardProfile>, profile: RewardProfile) {
    let signature = reward_profile_signature(&profile);
    if profiles
        .iter()
        .any(|existing| reward_profile_signature(existing) == signature)
    {
        return;
    }
    profiles.push(profile);
}

fn halton_unit(mut index: usize, base: usize) -> f64 {
    debug_assert!(base > 1);
    let mut scale = 1.0;
    let mut value = 0.0;
    while index > 0 {
        scale /= base as f64;
        value += scale * (index % base) as f64;
        index /= base;
    }
    value
}

fn centered_halton(index: usize, base: usize) -> f64 {
    halton_unit(index, base).mul_add(2.0, -1.0).clamp(-1.0, 1.0)
}

fn reward_profile_search_multiplier(index: usize, base: usize, radius: f64) -> f64 {
    1.0 + centered_halton(index, base) * radius
}

fn append_reward_profile_search_profiles(profiles: &mut Vec<RewardProfile>) {
    let samples = env_usize("SOCCER_CLIMB_REWARD_PROFILE_SEARCH_SAMPLES", 0).min(256);
    if samples == 0 || profiles.is_empty() {
        return;
    }

    let anchors = if env_bool("SOCCER_CLIMB_REWARD_PROFILE_SEARCH_AROUND_ALL", true) {
        profiles.clone()
    } else {
        profiles.iter().take(1).cloned().collect()
    };
    if anchors.is_empty() {
        return;
    }

    let radius = env_f64("SOCCER_CLIMB_REWARD_PROFILE_SEARCH_RADIUS", 0.18).clamp(0.0, 0.75);
    let quick_radius = env_f64(
        "SOCCER_CLIMB_REWARD_PROFILE_SEARCH_QUICK_RADIUS",
        radius * 1.25,
    )
    .clamp(0.0, 0.90);
    let shot_radius =
        env_f64("SOCCER_CLIMB_REWARD_PROFILE_SEARCH_SHOT_RADIUS", radius).clamp(0.0, 0.75);
    let commitment_radius =
        env_f64("SOCCER_CLIMB_REWARD_PROFILE_SEARCH_COMMIT_RADIUS", 0.06).clamp(0.0, 2.0);
    let start = env_usize("SOCCER_CLIMB_REWARD_PROFILE_SEARCH_SEED", 1).max(1);

    for sample_index in 0..samples {
        let anchor = &anchors[sample_index % anchors.len()];
        let sequence_index = start.saturating_add(sample_index / anchors.len() + 1);
        let profile = anchor.scaled(
            format!("search{}-{}", sample_index + 1, anchor.label),
            reward_profile_search_multiplier(sequence_index, 2, radius),
            reward_profile_search_multiplier(sequence_index, 3, radius),
            reward_profile_search_multiplier(sequence_index, 5, radius),
            reward_profile_search_multiplier(sequence_index, 7, quick_radius),
            reward_profile_search_multiplier(sequence_index, 11, shot_radius),
            centered_halton(sequence_index, 13) * commitment_radius,
        );
        push_unique_reward_profile(profiles, profile);
    }
}

fn climb_reward_profiles(base: RewardProfile) -> Vec<RewardProfile> {
    let mut profiles = std::env::var("SOCCER_CLIMB_REWARD_PROFILES")
        .ok()
        .into_iter()
        .flat_map(|raw| {
            raw.split('|')
                .enumerate()
                .filter_map(|(index, spec)| parse_reward_profile_spec(index + 1, spec))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    if profiles.is_empty() {
        profiles.push(base.clone());
        if env_bool("SOCCER_CLIMB_OPTIMIZE_REWARD_PROFILES", false) {
            let candidates = [
                ("build-complete", 0.96, 1.12, 1.05, 1.20, 1.00, 0.00),
                ("finish-light", 1.00, 1.00, 1.05, 1.00, 1.08, 0.05),
                ("field-balanced", 1.02, 1.08, 1.10, 1.10, 0.98, 0.02),
                ("central-finish", 0.94, 1.00, 1.05, 1.00, 1.10, 0.03),
                ("possession-safe", 0.90, 1.20, 1.20, 1.35, 0.90, 0.00),
                ("unlock-quick", 1.05, 1.00, 1.10, 1.50, 0.85, 0.00),
                ("clean-shot", 0.98, 1.05, 1.10, 1.10, 0.90, 0.01),
                ("late-box", 0.92, 1.18, 1.15, 1.45, 1.00, 0.02),
            ];
            let limit = env_usize("SOCCER_CLIMB_REWARD_PROFILE_LIMIT", candidates.len() + 1)
                .clamp(1, candidates.len() + 1);
            for (
                label,
                forward_pass,
                pass_chain,
                pass_turnover_penalty,
                quick_forward_release,
                shot_shaping,
                shot_commitment_add,
            ) in candidates.into_iter().take(limit.saturating_sub(1))
            {
                push_unique_reward_profile(
                    &mut profiles,
                    base.scaled(
                        label,
                        forward_pass,
                        pass_chain,
                        pass_turnover_penalty,
                        quick_forward_release,
                        shot_shaping,
                        shot_commitment_add,
                    ),
                );
            }
        }
    }
    append_reward_profile_search_profiles(&mut profiles);
    profiles
}

fn train_candidate_game(
    incoming_snapshot: Option<&SoccerNeuralNetworkSnapshot>,
    policy_shell: SoccerTeamQPolicies,
    neural: &SoccerNeuralLearningConfig,
    minutes: f64,
    period_count: usize,
    seed: u32,
    authoritative_lambda: f64,
    hermetic_neural: bool,
    vs_analytic: bool,
    candidate_home: bool,
) -> (
    Option<SoccerNeuralNetworkSnapshot>,
    Option<SoccerTeamQPolicies>,
) {
    let mut config = MatchConfig {
        duration_seconds: minutes * 60.0,
        period_count,
        learning_enabled: true,
        neural_learning: neural.clone(),
        seed,
        ..MatchConfig::default()
    };
    if hermetic_neural {
        // Keep a fresh empty Q-policy object only as a legal-action / plan
        // generator. The frontier itself is the neural snapshot; action-Q is
        // neither trained during the game nor carried between games.
        config.policy_train_max_transitions_per_tick = 0;
    }
    configure_authoritative_blend(&mut config, authoritative_lambda);
    let total_ticks = config.total_ticks();
    let mut sim = SoccerMatch::default_11v11(config).with_team_policies(policy_shell);
    if climb_uniform_elite_players_enabled() {
        sim.set_uniform_elite_players();
    }
    if vs_analytic {
        let (cand_team, opp_team) = if candidate_home {
            (Team::Home, Team::Away)
        } else {
            (Team::Away, Team::Home)
        };
        sim.disable_team_neural_brain(opp_team);
        if let Err(e) = sim.set_team_neural_brain(cand_team, incoming_snapshot.cloned(), false) {
            eprintln!("train: analytic-opponent candidate brain install failed: {e}");
        }
    } else if let Some(snapshot) = incoming_snapshot {
        if let Err(e) = sim.set_neural_network_snapshot(snapshot.clone()) {
            eprintln!("train: snapshot install failed: {e}");
        }
    }
    for _ in 0..total_ticks {
        sim.run_time_step();
    }
    sim.drain_neural_learning(Duration::from_millis(100));
    let next_snapshot = if vs_analytic {
        if candidate_home {
            sim.neural_network_snapshot_for(Team::Home)
        } else {
            sim.neural_network_snapshot_for(Team::Away)
        }
    } else {
        sim.neural_network_snapshot()
    };
    (next_snapshot, sim.team_policies().cloned())
}

#[allow(clippy::too_many_arguments)]
fn train_increment(
    mut snapshot: Option<SoccerNeuralNetworkSnapshot>,
    mut policies: SoccerTeamQPolicies,
    neural: &SoccerNeuralLearningConfig,
    increment_games: usize,
    cumulative: usize,
    minutes: f64,
    period_count: usize,
    seed_base: u32,
    authoritative_lambda: f64,
    hermetic_neural: bool,
    analytic_opponent_fraction: f64,
    alternate_training_sides: bool,
    paired_training_sides: bool,
    away_training_weight: usize,
    train_seed_variant_repeats: usize,
    seed_variant_mode: SeedVariantMode,
    training_side_mode: TrainingSideMode,
) -> (Option<SoccerNeuralNetworkSnapshot>, SoccerTeamQPolicies) {
    for g in 0..increment_games {
        let global_game = cumulative + g;
        let vs_analytic = analytic_opponent_game_enabled(global_game, analytic_opponent_fraction);
        let seed = seed_base.wrapping_add((cumulative + g) as u32);
        if vs_analytic && paired_training_sides && training_side_mode == TrainingSideMode::Paired {
            if seed_variant_mode == SeedVariantMode::Sequential && train_seed_variant_repeats > 1 {
                for variant_index in 0..train_seed_variant_repeats {
                    let mut trained_snapshots = Vec::new();
                    for (side_index, candidate_home) in [true, false].into_iter().enumerate() {
                        let policy_shell = if hermetic_neural {
                            SoccerTeamQPolicies::new(SoccerQPolicyOptions::default())
                        } else {
                            policies.clone()
                        };
                        let (next_snapshot, next_policies) = train_candidate_game(
                            snapshot.as_ref(),
                            policy_shell,
                            neural,
                            minutes,
                            period_count,
                            climb_variant_seed(seed, variant_index, side_index),
                            authoritative_lambda,
                            hermetic_neural,
                            true,
                            candidate_home,
                        );
                        if let Some(next_snapshot) = next_snapshot {
                            let weight = if candidate_home {
                                1
                            } else {
                                away_training_weight
                            };
                            for _ in 0..weight {
                                trained_snapshots.push(next_snapshot.clone());
                            }
                        }
                        if !hermetic_neural {
                            if let Some(next_policies) = next_policies {
                                policies = next_policies;
                            }
                        }
                    }
                    if let Some(averaged) = average_soccer_neural_network_snapshots(
                        snapshot.as_ref(),
                        &trained_snapshots,
                    ) {
                        snapshot = Some(averaged);
                    }
                }
            } else {
                let mut trained_snapshots = Vec::new();
                for variant_index in 0..train_seed_variant_repeats {
                    for (side_index, candidate_home) in [true, false].into_iter().enumerate() {
                        let policy_shell = if hermetic_neural {
                            SoccerTeamQPolicies::new(SoccerQPolicyOptions::default())
                        } else {
                            policies.clone()
                        };
                        let (next_snapshot, next_policies) = train_candidate_game(
                            snapshot.as_ref(),
                            policy_shell,
                            neural,
                            minutes,
                            period_count,
                            climb_variant_seed(seed, variant_index, side_index),
                            authoritative_lambda,
                            hermetic_neural,
                            true,
                            candidate_home,
                        );
                        if let Some(next_snapshot) = next_snapshot {
                            let weight = if candidate_home {
                                1
                            } else {
                                away_training_weight
                            };
                            for _ in 0..weight {
                                trained_snapshots.push(next_snapshot.clone());
                            }
                        }
                        if !hermetic_neural {
                            if let Some(next_policies) = next_policies {
                                policies = next_policies;
                            }
                        }
                    }
                }
                if let Some(averaged) =
                    average_soccer_neural_network_snapshots(snapshot.as_ref(), &trained_snapshots)
                {
                    snapshot = Some(averaged);
                }
            }
        } else if vs_analytic
            && matches!(
                training_side_mode,
                TrainingSideMode::HomeAway | TrainingSideMode::AwayHome
            )
        {
            let side_order: [bool; 2] = match training_side_mode {
                TrainingSideMode::HomeAway => [true, false],
                TrainingSideMode::AwayHome => [false, true],
                _ => unreachable!("guarded by matches! above"),
            };
            for variant_index in 0..train_seed_variant_repeats {
                for candidate_home in side_order {
                    let policy_shell = if hermetic_neural {
                        SoccerTeamQPolicies::new(SoccerQPolicyOptions::default())
                    } else {
                        policies.clone()
                    };
                    let side_index = if candidate_home { 0 } else { 1 };
                    let (next_snapshot, next_policies) = train_candidate_game(
                        snapshot.as_ref(),
                        policy_shell,
                        neural,
                        minutes,
                        period_count,
                        climb_variant_seed(seed, variant_index, side_index),
                        authoritative_lambda,
                        hermetic_neural,
                        true,
                        candidate_home,
                    );
                    if let Some(next_snapshot) = next_snapshot {
                        snapshot = Some(next_snapshot);
                    }
                    if !hermetic_neural {
                        if let Some(next_policies) = next_policies {
                            policies = next_policies;
                        }
                    }
                }
            }
        } else {
            let candidate_home = match training_side_mode {
                TrainingSideMode::Paired => !alternate_training_sides || global_game % 2 == 0,
                TrainingSideMode::HomeOnly => true,
                TrainingSideMode::AwayOnly => false,
                TrainingSideMode::HomeAway => true,
                TrainingSideMode::AwayHome => false,
            };
            for variant_index in 0..train_seed_variant_repeats {
                let policy_shell = if hermetic_neural {
                    SoccerTeamQPolicies::new(SoccerQPolicyOptions::default())
                } else {
                    policies.clone()
                };
                let (next_snapshot, next_policies) = train_candidate_game(
                    snapshot.as_ref(),
                    policy_shell,
                    neural,
                    minutes,
                    period_count,
                    climb_variant_seed(seed, variant_index, 0),
                    authoritative_lambda,
                    hermetic_neural,
                    vs_analytic,
                    candidate_home,
                );
                if let Some(next_snapshot) = next_snapshot {
                    snapshot = Some(next_snapshot);
                }
                if !hermetic_neural {
                    if let Some(next_policies) = next_policies {
                        policies = next_policies;
                    }
                }
            }
        }
    }
    (snapshot, policies)
}

#[derive(Clone, Debug, Default)]
struct EvalStats {
    payoff_sum: f64,
    games: u32,
    wins: u32,
    draws: u32,
    losses: u32,
    home_payoff_sum: f64,
    home_games: u32,
    home_wins: u32,
    home_draws: u32,
    home_losses: u32,
    home_goal_diff: i32,
    home_goals_for: u32,
    home_goals_against: u32,
    home_shots_for: u32,
    home_shots_against: u32,
    home_shots_on_target_for: u32,
    home_shots_on_target_against: u32,
    home_shots_after_pass_for: u32,
    home_forward_pass_margin: i32,
    home_net_forward_pass_margin: i32,
    away_payoff_sum: f64,
    away_games: u32,
    away_wins: u32,
    away_draws: u32,
    away_losses: u32,
    away_goal_diff: i32,
    away_goals_for: u32,
    away_goals_against: u32,
    away_shots_for: u32,
    away_shots_against: u32,
    away_shots_on_target_for: u32,
    away_shots_on_target_against: u32,
    away_shots_after_pass_for: u32,
    away_forward_pass_margin: i32,
    away_net_forward_pass_margin: i32,
    goal_diff: i32,
    goals_for: u32,
    goals_against: u32,
    shots_for: u32,
    shots_against: u32,
    shots_on_target_for: u32,
    shots_on_target_against: u32,
    shots_after_pass_for: u32,
    forward_passes_for: u32,
    forward_passes_against: u32,
    forward_pass_margin: i32,
    net_forward_pass_margin: i32,
}

impl EvalStats {
    fn mean_payoff(&self) -> f64 {
        if self.games > 0 {
            self.payoff_sum / self.games as f64
        } else {
            0.0
        }
    }

    fn mean_goal_diff(&self) -> f64 {
        if self.games > 0 {
            self.goal_diff as f64 / self.games as f64
        } else {
            0.0
        }
    }

    fn home_mean_payoff(&self) -> f64 {
        if self.home_games > 0 {
            self.home_payoff_sum / self.home_games as f64
        } else {
            0.0
        }
    }

    fn away_mean_payoff(&self) -> f64 {
        if self.away_games > 0 {
            self.away_payoff_sum / self.away_games as f64
        } else {
            0.0
        }
    }

    fn weak_side_mean_payoff(&self) -> f64 {
        match (self.home_games, self.away_games) {
            (0, 0) => 0.0,
            (0, _) => self.away_mean_payoff(),
            (_, 0) => self.home_mean_payoff(),
            _ => self.home_mean_payoff().min(self.away_mean_payoff()),
        }
    }

    fn climb_potential_score(&self) -> f64 {
        let games = self.games.max(1) as f64;
        let shots_on_target_margin =
            self.shots_on_target_for as f64 - self.shots_on_target_against as f64;
        self.weak_side_mean_payoff() * 1.20
            + self.mean_goal_diff() * 0.60
            + (shots_on_target_margin / games) * 0.15
            + (self.shots_after_pass_for as f64 / games) * 0.10
            + (self.net_forward_pass_margin as f64 / games) * 0.05
            + (self.wins as f64 / games) * 0.10
            - (self.losses as f64 / games) * 0.05
    }

    fn add_assign(&mut self, other: &EvalStats) {
        self.payoff_sum += other.payoff_sum;
        self.games = self.games.saturating_add(other.games);
        self.wins = self.wins.saturating_add(other.wins);
        self.draws = self.draws.saturating_add(other.draws);
        self.losses = self.losses.saturating_add(other.losses);
        self.home_payoff_sum += other.home_payoff_sum;
        self.home_games = self.home_games.saturating_add(other.home_games);
        self.home_wins = self.home_wins.saturating_add(other.home_wins);
        self.home_draws = self.home_draws.saturating_add(other.home_draws);
        self.home_losses = self.home_losses.saturating_add(other.home_losses);
        self.home_goal_diff = self.home_goal_diff.saturating_add(other.home_goal_diff);
        self.home_goals_for = self.home_goals_for.saturating_add(other.home_goals_for);
        self.home_goals_against = self
            .home_goals_against
            .saturating_add(other.home_goals_against);
        self.home_shots_for = self.home_shots_for.saturating_add(other.home_shots_for);
        self.home_shots_against = self
            .home_shots_against
            .saturating_add(other.home_shots_against);
        self.home_shots_on_target_for = self
            .home_shots_on_target_for
            .saturating_add(other.home_shots_on_target_for);
        self.home_shots_on_target_against = self
            .home_shots_on_target_against
            .saturating_add(other.home_shots_on_target_against);
        self.home_shots_after_pass_for = self
            .home_shots_after_pass_for
            .saturating_add(other.home_shots_after_pass_for);
        self.home_forward_pass_margin = self
            .home_forward_pass_margin
            .saturating_add(other.home_forward_pass_margin);
        self.home_net_forward_pass_margin = self
            .home_net_forward_pass_margin
            .saturating_add(other.home_net_forward_pass_margin);
        self.away_payoff_sum += other.away_payoff_sum;
        self.away_games = self.away_games.saturating_add(other.away_games);
        self.away_wins = self.away_wins.saturating_add(other.away_wins);
        self.away_draws = self.away_draws.saturating_add(other.away_draws);
        self.away_losses = self.away_losses.saturating_add(other.away_losses);
        self.away_goal_diff = self.away_goal_diff.saturating_add(other.away_goal_diff);
        self.away_goals_for = self.away_goals_for.saturating_add(other.away_goals_for);
        self.away_goals_against = self
            .away_goals_against
            .saturating_add(other.away_goals_against);
        self.away_shots_for = self.away_shots_for.saturating_add(other.away_shots_for);
        self.away_shots_against = self
            .away_shots_against
            .saturating_add(other.away_shots_against);
        self.away_shots_on_target_for = self
            .away_shots_on_target_for
            .saturating_add(other.away_shots_on_target_for);
        self.away_shots_on_target_against = self
            .away_shots_on_target_against
            .saturating_add(other.away_shots_on_target_against);
        self.away_shots_after_pass_for = self
            .away_shots_after_pass_for
            .saturating_add(other.away_shots_after_pass_for);
        self.away_forward_pass_margin = self
            .away_forward_pass_margin
            .saturating_add(other.away_forward_pass_margin);
        self.away_net_forward_pass_margin = self
            .away_net_forward_pass_margin
            .saturating_add(other.away_net_forward_pass_margin);
        self.goal_diff = self.goal_diff.saturating_add(other.goal_diff);
        self.goals_for = self.goals_for.saturating_add(other.goals_for);
        self.goals_against = self.goals_against.saturating_add(other.goals_against);
        self.shots_for = self.shots_for.saturating_add(other.shots_for);
        self.shots_against = self.shots_against.saturating_add(other.shots_against);
        self.shots_on_target_for = self
            .shots_on_target_for
            .saturating_add(other.shots_on_target_for);
        self.shots_on_target_against = self
            .shots_on_target_against
            .saturating_add(other.shots_on_target_against);
        self.shots_after_pass_for = self
            .shots_after_pass_for
            .saturating_add(other.shots_after_pass_for);
        self.forward_passes_for = self
            .forward_passes_for
            .saturating_add(other.forward_passes_for);
        self.forward_passes_against = self
            .forward_passes_against
            .saturating_add(other.forward_passes_against);
        self.forward_pass_margin = self
            .forward_pass_margin
            .saturating_add(other.forward_pass_margin);
        self.net_forward_pass_margin = self
            .net_forward_pass_margin
            .saturating_add(other.net_forward_pass_margin);
    }
}

fn eval_is_better(candidate: &EvalStats, incumbent: Option<&EvalStats>) -> bool {
    let Some(incumbent) = incumbent else {
        return true;
    };
    let candidate_score = candidate.mean_payoff();
    let incumbent_score = incumbent.mean_payoff();
    const EPS: f64 = 1.0e-9;
    if candidate_score > incumbent_score + EPS {
        return true;
    }
    if candidate_score + EPS < incumbent_score {
        return false;
    }
    let candidate_potential = candidate.climb_potential_score();
    let incumbent_potential = incumbent.climb_potential_score();
    if candidate_potential > incumbent_potential + EPS {
        return true;
    }
    if candidate_potential + EPS < incumbent_potential {
        return false;
    }
    let candidate_weak_side = candidate.weak_side_mean_payoff();
    let incumbent_weak_side = incumbent.weak_side_mean_payoff();
    if candidate_weak_side > incumbent_weak_side + EPS {
        return true;
    }
    if candidate_weak_side + EPS < incumbent_weak_side {
        return false;
    }
    if candidate.goal_diff != incumbent.goal_diff {
        return candidate.goal_diff > incumbent.goal_diff;
    }
    if candidate.wins != incumbent.wins {
        return candidate.wins > incumbent.wins;
    }
    let candidate_away = candidate.away_mean_payoff();
    let incumbent_away = incumbent.away_mean_payoff();
    if candidate_away > incumbent_away + EPS {
        return true;
    }
    if candidate_away + EPS < incumbent_away {
        return false;
    }
    if candidate.goals_against != incumbent.goals_against {
        return candidate.goals_against < incumbent.goals_against;
    }
    if candidate.losses != incumbent.losses {
        return candidate.losses < incumbent.losses;
    }
    candidate.net_forward_pass_margin > incumbent.net_forward_pass_margin
}

struct CandidateResult {
    learning_rate: f64,
    authoritative_lambda: f64,
    seed_variant_mode: SeedVariantMode,
    training_side_mode: TrainingSideMode,
    reward_profile_label: String,
    snapshot: SoccerNeuralNetworkSnapshot,
    policies: SoccerTeamQPolicies,
    eval: EvalStats,
}

/// Head-to-head payoff of a FROZEN candidate net vs the pure-analytic engine over `games` matches,
/// candidate alternating Home/Away for fairness. Payoff = (wins + 0.5·draws) / games.
fn eval_vs_analytic_game(
    snapshot: &SoccerNeuralNetworkSnapshot,
    game_index: usize,
    minutes: f64,
    period_count: usize,
    seed_base: u32,
    authoritative_lambda: f64,
) -> EvalStats {
    let mut stats = EvalStats::default();
    let candidate_home = game_index % 2 == 0;
    // Inference-only: neural ON (so the net drives), learning OFF (frozen — no target drift).
    let mut config = MatchConfig {
        duration_seconds: minutes * 60.0,
        period_count,
        learning_enabled: false,
        neural_learning: climb_neural_config(0.015),
        // Pair each scenario across both orientations. Using `game_index` directly
        // makes every Home candidate consume an even seed and every Away candidate
        // an odd seed, confounding side strength with seed parity.
        seed: eval_scenario_seed(seed_base, game_index),
        ..MatchConfig::default()
    };
    configure_authoritative_blend(&mut config, authoritative_lambda);
    let total_ticks = config.total_ticks();
    let mut sim = SoccerMatch::default_11v11(config)
        .with_team_policies(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
    if climb_uniform_elite_players_enabled() {
        sim.set_uniform_elite_players();
    }
    // Candidate = frozen net on one side; the other side stays on the analytic decision stack.
    let (cand_team, opp_team) = if candidate_home {
        (Team::Home, Team::Away)
    } else {
        (Team::Away, Team::Home)
    };
    // Disable the analytic side explicitly, then install the candidate's complete frozen brain.
    // Per-team actor sidecars keep either orientation isolated and equivalent.
    sim.disable_team_neural_brain(opp_team);
    if let Err(e) = sim.set_team_neural_brain(cand_team, Some(snapshot.clone()), true) {
        eprintln!("eval game {game_index}: candidate brain install failed: {e}");
        return stats;
    }
    for _ in 0..total_ticks {
        sim.run_time_step();
    }
    let summary = sim.summary();
    let (cand_goals, opp_goals) = if candidate_home {
        (summary.score_home, summary.score_away)
    } else {
        (summary.score_away, summary.score_home)
    };
    let payoff = if cand_goals > opp_goals {
        stats.wins += 1;
        1.0
    } else if cand_goals == opp_goals {
        stats.draws += 1;
        0.5
    } else {
        stats.losses += 1;
        0.0
    };
    let match_stats = &summary.stats;
    let (
        shots_for,
        shots_against,
        sot_for,
        sot_against,
        shots_after_pass_for,
        forward_for,
        forward_against,
        net_forward_margin,
    ) = if candidate_home {
        (
            match_stats.shots_home,
            match_stats.shots_away,
            match_stats.shots_on_target_home,
            match_stats.shots_on_target_away,
            match_stats.shots_after_pass_home,
            match_stats.passes_completed_forward_home,
            match_stats.passes_completed_forward_away,
            match_stats.passes_completed_forward_home as i32
                - match_stats.interceptions_away as i32
                - (match_stats.passes_completed_forward_away as i32
                    - match_stats.interceptions_home as i32),
        )
    } else {
        (
            match_stats.shots_away,
            match_stats.shots_home,
            match_stats.shots_on_target_away,
            match_stats.shots_on_target_home,
            match_stats.shots_after_pass_away,
            match_stats.passes_completed_forward_away,
            match_stats.passes_completed_forward_home,
            match_stats.passes_completed_forward_away as i32
                - match_stats.interceptions_home as i32
                - (match_stats.passes_completed_forward_home as i32
                    - match_stats.interceptions_away as i32),
        )
    };
    stats.payoff_sum += payoff;
    stats.games += 1;
    if candidate_home {
        stats.home_payoff_sum += payoff;
        stats.home_games += 1;
        if payoff >= 1.0 {
            stats.home_wins += 1;
        } else if payoff > 0.0 {
            stats.home_draws += 1;
        } else {
            stats.home_losses += 1;
        }
        stats.home_goal_diff += cand_goals as i32 - opp_goals as i32;
        stats.home_goals_for += cand_goals;
        stats.home_goals_against += opp_goals;
        stats.home_shots_for += shots_for;
        stats.home_shots_against += shots_against;
        stats.home_shots_on_target_for += sot_for;
        stats.home_shots_on_target_against += sot_against;
        stats.home_shots_after_pass_for += shots_after_pass_for;
        stats.home_forward_pass_margin += forward_for as i32 - forward_against as i32;
        stats.home_net_forward_pass_margin += net_forward_margin;
    } else {
        stats.away_payoff_sum += payoff;
        stats.away_games += 1;
        if payoff >= 1.0 {
            stats.away_wins += 1;
        } else if payoff > 0.0 {
            stats.away_draws += 1;
        } else {
            stats.away_losses += 1;
        }
        stats.away_goal_diff += cand_goals as i32 - opp_goals as i32;
        stats.away_goals_for += cand_goals;
        stats.away_goals_against += opp_goals;
        stats.away_shots_for += shots_for;
        stats.away_shots_against += shots_against;
        stats.away_shots_on_target_for += sot_for;
        stats.away_shots_on_target_against += sot_against;
        stats.away_shots_after_pass_for += shots_after_pass_for;
        stats.away_forward_pass_margin += forward_for as i32 - forward_against as i32;
        stats.away_net_forward_pass_margin += net_forward_margin;
    }
    stats.goal_diff += cand_goals as i32 - opp_goals as i32;
    stats.goals_for += cand_goals;
    stats.goals_against += opp_goals;
    stats.shots_for += shots_for;
    stats.shots_against += shots_against;
    stats.shots_on_target_for += sot_for;
    stats.shots_on_target_against += sot_against;
    stats.shots_after_pass_for += shots_after_pass_for;
    stats.forward_passes_for += forward_for;
    stats.forward_passes_against += forward_against;
    stats.forward_pass_margin += forward_for as i32 - forward_against as i32;
    stats.net_forward_pass_margin += net_forward_margin;
    stats
}

fn eval_scenario_seed(seed_base: u32, game_index: usize) -> u32 {
    seed_base.wrapping_add((game_index / 2) as u32)
}

fn eval_vs_analytic(
    snapshot: &SoccerNeuralNetworkSnapshot,
    games: usize,
    minutes: f64,
    period_count: usize,
    seed_base: u32,
    authoritative_lambda: f64,
) -> EvalStats {
    let parallelism = env_usize("SOCCER_CLIMB_EVAL_PARALLELISM", 1).clamp(1, games.max(1));
    if parallelism <= 1 || games <= 1 {
        let mut stats = EvalStats::default();
        for game_index in 0..games {
            let fixture = eval_vs_analytic_game(
                snapshot,
                game_index,
                minutes,
                period_count,
                seed_base,
                authoritative_lambda,
            );
            stats.add_assign(&fixture);
        }
        return stats;
    }

    let next_game = AtomicUsize::new(0);
    let results = Mutex::new(vec![None; games]);
    std::thread::scope(|scope| {
        for _ in 0..parallelism {
            scope.spawn(|| loop {
                let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                if game_index >= games {
                    break;
                }
                let fixture = eval_vs_analytic_game(
                    snapshot,
                    game_index,
                    minutes,
                    period_count,
                    seed_base,
                    authoritative_lambda,
                );
                if let Ok(mut guard) = results.lock() {
                    guard[game_index] = Some(fixture);
                }
            });
        }
    });
    let mut stats = EvalStats::default();
    if let Ok(guard) = results.into_inner() {
        for fixture in guard.into_iter().flatten() {
            stats.add_assign(&fixture);
        }
    }
    stats
}

fn main() {
    // Deterministic formation LP so a given seed reproduces (fair increment-to-increment comparison).
    enable_deterministic_formation_lp();
    env_default_bool("DD_SOCCER_ENABLE_TARGET_STANDARDIZATION", true);
    let dp_bootstrap = env_default_bool("DD_SOCCER_ENABLE_DP_BOOTSTRAP", true);
    let dp_critic_target = env_default_bool("DD_SOCCER_ENABLE_DP_CRITIC_TARGET", true);
    let dp_bootstrap_horizon = env_default_usize("DD_SOCCER_DP_BOOTSTRAP_HORIZON", 64).max(1);
    let dp_bootstrap_sweeps = env_default_usize("DD_SOCCER_DP_BOOTSTRAP_SWEEPS", 200).max(1);
    let approx_dp_replay_passes = env_default_usize("SOCCER_APPROX_DP_REPLAY_PASSES", 8).max(1);
    let climb_mpc_enabled = env_default_bool("SOCCER_CLIMB_MPC_ENABLED", true);
    let climb_mpc_field_aware = env_default_bool("SOCCER_CLIMB_MPC_FIELD_AWARE_ENABLED", true);
    let climb_mpc_reconcile = env_default_bool("SOCCER_CLIMB_MPC_RECONCILE_ENABLED", true);
    let climb_mpc_latent = env_default_bool("SOCCER_CLIMB_MPC_LATENT_OBJECTIVE_ENABLED", true);
    let climb_mpc_horizon = env_default_usize("SOCCER_CLIMB_MPC_PLAYER_HORIZON", 30).max(1);
    env_default_bool("DD_SOCCER_ENABLE_MC_CRITIC_TARGET", true);
    env_default_bool("DD_SOCCER_ENABLE_NEURAL_SELF_BOOTSTRAP", true);
    env_default_bool("DD_SOCCER_ENABLE_MAXA_BOOTSTRAP", true);
    env_default_bool("DD_SOCCER_ENABLE_NOVELTY_BONUS", true);
    env_default_bool("DD_SOCCER_ENABLE_FAST_ACTOR_REWARD_FALLBACK", true);
    env_default_bool("DD_SOCCER_ENABLE_PASS_OUTCOME_PRIORITY_LEARNING", true);
    env_default_bool("DD_SOCCER_FORWARD_PASS_CLIMB_CURRICULUM", true);
    let planner_teacher_missed_opportunity =
        env_default_bool("DD_SOCCER_ENABLE_PLANNER_TEACHER_MISSED_OPPORTUNITY", true);
    let planner_teacher_advantage_floor =
        env_default_f64("SOCCER_PLANNER_TEACHER_ADVANTAGE_FLOOR", 0.045);
    let planner_teacher_advantage_max =
        env_default_f64("SOCCER_PLANNER_TEACHER_ADVANTAGE_MAX", 0.30);
    let planner_teacher_weight = env_default_f64("SOCCER_PLANNER_TEACHER_WEIGHT", 1.8);
    let planner_teacher_min_score_share =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_SCORE_SHARE", 0.006);
    let planner_teacher_min_shot_quality =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_SHOT_QUALITY", 0.52);
    let planner_teacher_min_forward_quality =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_FORWARD_QUALITY", 0.50);
    let planner_teacher_min_forward_completion =
        env_default_f64("SOCCER_PLANNER_TEACHER_MIN_FORWARD_COMPLETION", 0.38);
    let planner_teacher_max_samples_per_decision =
        env_default_usize("SOCCER_PLANNER_TEACHER_MAX_SAMPLES_PER_DECISION", 2);
    let planner_teacher_include_same_pass_family =
        env_default_bool("DD_SOCCER_PLANNER_TEACHER_INCLUDE_SAME_PASS_FAMILY", true);
    let planner_teacher_same_pass_min_margin_share =
        env_default_f64("SOCCER_PLANNER_TEACHER_SAME_PASS_MIN_MARGIN_SHARE", 0.015);
    let attack_canonical_features =
        env_default_bool("DD_SOCCER_ENABLE_ATTACK_CANONICAL_NEURAL_FEATURES", true);
    let ball_zone_tactical_scale =
        env_default_bool("DD_SOCCER_ENABLE_BALL_ZONE_TACTICAL_SCALE", true);
    let player_grid_xy_features =
        env_default_bool("DD_SOCCER_ENABLE_PLAYER_GRID_XY_FEATURES", true);
    let uniform_elite_players = env_default_bool("SOCCER_ENGINE_UNIFORM_ELITE_PLAYERS", true);
    let seed_varied_skills = env_default_bool("SOCCER_SEED_VARIED_SKILLS", !uniform_elite_players);
    env_default_bool("DD_SOCCER_ENABLE_FORWARD_RELEASE_ROOT_CANDIDATE", true);
    env_default_f64("DD_SOCCER_FORWARD_RELEASE_MIN_QUALITY", 0.35);
    env_default_f64("DD_SOCCER_FORWARD_RELEASE_MIN_COMPLETION", 0.35);
    env_default_bool("DD_SOCCER_ENABLE_FORWARD_SELECT_LOGIT", true);
    let forward_select_logit_weight = env_default_f64("DD_SOCCER_FORWARD_SELECT_LOGIT_WEIGHT", 2.0);
    let finishing_select_bonus_weight =
        env_default_f64("SOCCER_NEURAL_FINISHING_SELECT_BONUS_WEIGHT", 0.70);
    let finishing_selection_floor =
        env_default_f64("SOCCER_NEURAL_FINISHING_SELECTION_FLOOR", 0.30);
    let finishing_selection_max_regression = env_default_f64(
        "SOCCER_NEURAL_FINISHING_SELECTION_MAX_SCORE_REGRESSION",
        12.0,
    );
    let finishing_selection_shot_drought_boost =
        env_default_f64("SOCCER_NEURAL_FINISHING_SELECTION_SHOT_DROUGHT_BOOST", 0.0);
    let finishing_selection_max_effective_floor = env_default_f64(
        "SOCCER_NEURAL_FINISHING_SELECTION_MAX_EFFECTIVE_FLOOR",
        0.52,
    );
    let forward_creation_selection_floor =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_SELECTION_FLOOR", 0.22);
    let forward_creation_selection_max_regression = env_default_f64(
        "SOCCER_NEURAL_FORWARD_CREATION_SELECTION_MAX_SCORE_REGRESSION",
        18.0,
    );
    let forward_creation_min_quality =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_MIN_QUALITY", 0.50);
    let forward_creation_min_completion =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_MIN_COMPLETION", 0.45);
    let forward_creation_min_forward_yards =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_MIN_FORWARD_YARDS", 4.0);
    let forward_creation_bypass_min_quality =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_BYPASS_MIN_QUALITY", 0.70);
    let forward_creation_bypass_min_completion =
        env_default_f64("SOCCER_NEURAL_FORWARD_CREATION_BYPASS_MIN_COMPLETION", 0.48);
    env_default_usize("SOCCER_NEURAL_MCTS_PASS_TARGET_CANDIDATES", 6);
    env_default_bool("DD_SOCCER_ENABLE_OVERLOAD_WEIGHTED_PROGRESSION", true);
    let forward_pass_reward_scale = env_default_f64("DD_SOCCER_FORWARD_PASS_REWARD_SCALE", 6.25);
    let pass_chain_reward_scale = env_default_f64("DD_SOCCER_PASS_CHAIN_REWARD_SCALE", 4.25);
    let pass_turnover_penalty_scale = env_default_f64("DD_SOCCER_PASS_TURNOVER_PENALTY_SCALE", 5.0);
    let quick_forward_release_reward_scale =
        env_default_f64("DD_SOCCER_QUICK_FORWARD_RELEASE_REWARD_SCALE", 1.25);
    let shot_shaping_reward_scale = env_default_f64("DD_SOCCER_SHOT_SHAPING_REWARD_SCALE", 0.85);
    let shot_commitment_reward_scale =
        env_default_f64("DD_SOCCER_SHOT_COMMITMENT_REWARD_SCALE", 0.0);
    let base_reward_profile = RewardProfile {
        label: "base".to_string(),
        forward_pass: forward_pass_reward_scale,
        pass_chain: pass_chain_reward_scale,
        pass_turnover_penalty: pass_turnover_penalty_scale,
        quick_forward_release: quick_forward_release_reward_scale,
        shot_shaping: shot_shaping_reward_scale,
        shot_commitment: shot_commitment_reward_scale,
    };
    let reward_profiles = climb_reward_profiles(base_reward_profile.clone());
    if reward_profiles.len() > 1 {
        std::env::set_var("SOCCER_DYNAMIC_REWARD_WEIGHTS", "1");
    }
    let dynamic_reward_weights = env_bool("SOCCER_DYNAMIC_REWARD_WEIGHTS", false);
    let reward_profile_summaries = reward_profiles
        .iter()
        .map(RewardProfile::summary)
        .collect::<Vec<_>>();
    let chance_quality_reward = env_default_bool("DD_SOCCER_ENABLE_CHANCE_QUALITY_REWARD", true);
    let chance_quality_composite =
        env_default_bool("DD_SOCCER_ENABLE_CHANCE_QUALITY_COMPOSITE", true);
    let chance_quality_k = env_default_f64("DD_SOCCER_CHANCE_QUALITY_K", 22.0);
    let chance_quality_cap = env_default_f64("DD_SOCCER_CHANCE_QUALITY_CAP", 5.0);
    let authoritative_lambda = env_default_f64("DD_SOCCER_NEURAL_AUTHORITATIVE_LAMBDA", 8.0);
    let authoritative_lambda_candidates =
        climb_authoritative_lambda_candidates(authoritative_lambda);
    let policy_entropy_coeff = env_default_f64("SOCCER_POLICY_ENTROPY_COEFF", 0.04);
    let neural_mcts_enabled = env_bool("SOCCER_CLIMB_NEURAL_MCTS_ENABLED", false)
        && std::env::var("SOCCER_DISABLE_NEURAL_MCTS").ok().as_deref() != Some("1");
    let neural_mcts_simulations = env_usize_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_SIMULATIONS",
            "SOCCER_MCTS_SIMULATIONS",
        ],
        8,
    )
    .clamp(1, 32);
    let neural_mcts_candidates = env_usize_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_CANDIDATES",
            "SOCCER_MCTS_CANDIDATES",
        ],
        4,
    )
    .clamp(2, 16);
    let neural_mcts_depth =
        env_usize_any(&["SOCCER_CLIMB_NEURAL_MCTS_DEPTH", "SOCCER_MCTS_DEPTH"], 1).clamp(1, 3);
    let neural_mcts_exploration = env_f64_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_EXPLORATION",
            "SOCCER_MCTS_EXPLORATION",
        ],
        1.25,
    )
    .clamp(0.0, 4.0);
    let neural_mcts_model_weight = env_f64_any(
        &[
            "SOCCER_CLIMB_NEURAL_MCTS_MODEL_WEIGHT",
            "SOCCER_MCTS_MODEL_WEIGHT",
        ],
        0.35,
    )
    .clamp(0.0, 1.0);

    let args: Vec<String> = std::env::args().collect();
    let increment_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let minutes: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let eval_games: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let increments: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
    let cli_seed_base: u32 = parse_seed_arg(args.get(5), 0x5EED_0000);
    let period_count = env_default_usize("SOCCER_CLIMB_PERIOD_COUNT", 1).clamp(1, 4);
    let seed_bases = climb_seed_bases(cli_seed_base);
    let learning_rate_candidates = climb_learning_rate_candidates();
    let neural = climb_neural_config(learning_rate_candidates[0]);
    let analytic_opponent_fraction =
        env_f64("SOCCER_CLIMB_ANALYTIC_OPPONENT_FRAC", 1.0).clamp(0.0, 1.0);
    let hermetic_neural = env_default_bool("SOCCER_CLIMB_HERMETIC_NEURAL", true);
    let alternate_training_sides = env_default_bool("SOCCER_CLIMB_TRAIN_ALTERNATE_SIDES", false);
    let paired_training_sides = env_default_bool(
        "SOCCER_CLIMB_PAIRED_TRAINING_SIDES",
        hermetic_neural && !alternate_training_sides,
    );
    let away_training_weight =
        env_default_usize("SOCCER_CLIMB_AWAY_TRAINING_WEIGHT", 1).clamp(1, 4);
    let train_seed_variant_repeats_default = if uniform_elite_players && !seed_varied_skills {
        1
    } else {
        2
    };
    let train_seed_variant_repeats = env_default_usize(
        "SOCCER_CLIMB_TRAIN_SEED_VARIANTS",
        train_seed_variant_repeats_default,
    )
    .clamp(1, 8);
    let varied_skill_curriculum =
        train_seed_variant_repeats > 1 && (!uniform_elite_players || seed_varied_skills);
    let seed_variant_modes = climb_seed_variant_modes(false);
    let seed_variant_mode_labels = seed_variant_modes
        .iter()
        .map(|mode| mode.label())
        .collect::<Vec<_>>();
    let training_side_modes = climb_training_side_modes(
        varied_skill_curriculum && env_bool("SOCCER_CLIMB_OPTIMIZE_SIDE_MODES", false),
        paired_training_sides,
        alternate_training_sides,
    );
    let training_side_mode_labels = training_side_modes
        .iter()
        .map(|mode| mode.label())
        .collect::<Vec<_>>();

    let mut aggregate_eval = EvalStats::default();
    let mut aggregate_seed_count = 0u32;

    for (seed_index, seed_base) in seed_bases.iter().copied().enumerate() {
        let eval_seed_base = seed_base ^ 0xE7A1_0000;

        println!("===== CLIMB RATCHET (keep-best; monotonic by construction) =====");
        println!(
            "increment_games={increment_games} minutes={minutes} eval_games={eval_games} \
         increments={increments} period_count={period_count} seed_base=0x{seed_base:08X} seed_index={} seed_count={}",
            seed_index + 1,
            seed_bases.len()
        );
        println!(
         "mode=neural-authoritative lambda_candidates={:?} analytic_train_frac={analytic_opponent_fraction:.2} \
          hermetic_neural={hermetic_neural} alternate_training_sides={alternate_training_sides} \
          paired_training_sides={paired_training_sides} away_training_weight={away_training_weight} \
          train_seed_variants={train_seed_variant_repeats} seed_variant_modes={:?} train_side_modes={:?} \
          reward_profiles={:?} dynamic_reward_weights={dynamic_reward_weights} \
          uniform_elite_players={uniform_elite_players} seed_varied_skills={seed_varied_skills} \
          attack_canonical_features={attack_canonical_features} ball_zone_tactical_scale={ball_zone_tactical_scale} \
         player_grid_xy_features={player_grid_xy_features} \
         hidden={} lr_candidates={:?} momentum={} target_scale={} popart={} entropy_coeff={policy_entropy_coeff} \
         mappo_team_share={} marl_team_weight={} marl_intermediate_weight={} mappo_clip={} \
         forward_select_logit_weight={forward_select_logit_weight} finishing_select_bonus_weight={finishing_select_bonus_weight} \
         finishing_selection_floor={finishing_selection_floor} finishing_selection_max_regression={finishing_selection_max_regression} \
         finishing_selection_shot_drought_boost={finishing_selection_shot_drought_boost} \
         finishing_selection_max_effective_floor={finishing_selection_max_effective_floor} \
         forward_creation_selection_floor={forward_creation_selection_floor} \
         forward_creation_selection_max_regression={forward_creation_selection_max_regression} \
         forward_creation_min_quality={forward_creation_min_quality} \
         forward_creation_min_completion={forward_creation_min_completion} \
         forward_creation_min_forward_yards={forward_creation_min_forward_yards} \
         forward_creation_bypass_min_quality={forward_creation_bypass_min_quality} \
         forward_creation_bypass_min_completion={forward_creation_bypass_min_completion} \
         planner_teacher_missed_opportunity={planner_teacher_missed_opportunity} \
         planner_teacher_advantage_floor={planner_teacher_advantage_floor:.3} \
         planner_teacher_advantage_max={planner_teacher_advantage_max:.3} \
         planner_teacher_weight={planner_teacher_weight:.2} \
         planner_teacher_min_score_share={planner_teacher_min_score_share:.3} \
         planner_teacher_min_shot_quality={planner_teacher_min_shot_quality:.2} \
         planner_teacher_min_forward_quality={planner_teacher_min_forward_quality:.2} \
         planner_teacher_min_forward_completion={planner_teacher_min_forward_completion:.2} \
         planner_teacher_max_samples_per_decision={planner_teacher_max_samples_per_decision} \
         planner_teacher_include_same_pass_family={planner_teacher_include_same_pass_family} \
         planner_teacher_same_pass_min_margin_share={planner_teacher_same_pass_min_margin_share:.3} \
         neural_mcts_enabled={neural_mcts_enabled} neural_mcts_simulations={neural_mcts_simulations} \
        neural_mcts_candidates={neural_mcts_candidates} neural_mcts_depth={neural_mcts_depth} \
         neural_mcts_exploration={neural_mcts_exploration:.2} neural_mcts_model_weight={neural_mcts_model_weight:.2}",
        authoritative_lambda_candidates,
        seed_variant_mode_labels,
        training_side_mode_labels,
        reward_profile_summaries,
        neural.hidden_units,
        learning_rate_candidates,
        neural.optimizer_momentum,
        neural.target_scale,
        neural.target_popart_enabled,
        neural.mappo_team_reward_share,
        neural.marl_team_reward_weight,
        neural.marl_intermediate_reward_weight,
        neural.mappo_clip_epsilon,
    );
        println!(
        "base_reward_profile forward_pass={forward_pass_reward_scale:.2} pass_chain={pass_chain_reward_scale:.2} \
         pass_turnover_penalty={pass_turnover_penalty_scale:.2} quick_forward_release={quick_forward_release_reward_scale:.2} \
         shot_shaping={shot_shaping_reward_scale:.2} shot_commitment={shot_commitment_reward_scale:.2} \
         chance_quality_reward={chance_quality_reward} \
         chance_quality_composite={chance_quality_composite} chance_quality_k={chance_quality_k:.2} \
         chance_quality_cap={chance_quality_cap:.2}"
    );
        println!(
            "execution_stack dp_bootstrap={dp_bootstrap} dp_critic_target={dp_critic_target} \
             dp_horizon={dp_bootstrap_horizon} dp_sweeps={dp_bootstrap_sweeps} \
             approx_dp_replay_passes={approx_dp_replay_passes} lp_coupling={} \
             mpc={climb_mpc_enabled} field_aware={} reconcile={} latent={} horizon={climb_mpc_horizon}",
            neural.lp_coupling_enabled,
            climb_mpc_enabled && climb_mpc_field_aware,
            climb_mpc_enabled && climb_mpc_reconcile,
            climb_mpc_enabled && climb_mpc_latent,
        );
        println!(
        "(eval = frozen candidate net vs the pure-analytic engine; payoff > 0.5 ⇒ beats analytic)"
    );

        let mut policies = SoccerTeamQPolicies::new(SoccerQPolicyOptions::default());
        let mut snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
        let mut best_snapshot: Option<SoccerNeuralNetworkSnapshot> = None;
        let mut best_policies: Option<SoccerTeamQPolicies> = None;
        let mut best_eval: Option<EvalStats> = None;
        let mut best_at_games = 0usize;
        let mut cumulative = 0usize;

        for inc in 0..increments {
            let start_snapshot = snapshot.clone();
            let start_policies = policies.clone();
            cumulative += increment_games;

            let mut increment_best: Option<CandidateResult> = None;
            let mut candidate_index = 0usize;
            for authoritative_lambda in authoritative_lambda_candidates.iter().copied() {
                for learning_rate in learning_rate_candidates.iter().copied() {
                    for reward_profile in &reward_profiles {
                        for seed_variant_mode in seed_variant_modes.iter().copied() {
                            for training_side_mode in training_side_modes.iter().copied() {
                                reward_profile.apply();
                                let neural = climb_neural_config(learning_rate);
                                let (candidate_snapshot, candidate_policies) = train_increment(
                                    start_snapshot.clone(),
                                    start_policies.clone(),
                                    &neural,
                                    increment_games,
                                    cumulative - increment_games,
                                    minutes,
                                    period_count,
                                    seed_base,
                                    authoritative_lambda,
                                    hermetic_neural,
                                    analytic_opponent_fraction,
                                    alternate_training_sides,
                                    paired_training_sides,
                                    away_training_weight,
                                    train_seed_variant_repeats,
                                    seed_variant_mode,
                                    training_side_mode,
                                );
                                let Some(candidate_snapshot) = candidate_snapshot else {
                                    println!(
                    "increment {inc:>2} cand={candidate_index} lr={learning_rate} lambda={authoritative_lambda} reward_profile={} variant_mode={} train_side={}: no snapshot yet",
                    reward_profile.label,
                    seed_variant_mode.label(),
                    training_side_mode.label()
                );
                                    candidate_index = candidate_index.saturating_add(1);
                                    continue;
                                };
                                let eval = eval_vs_analytic(
                                    &candidate_snapshot,
                                    eval_games,
                                    minutes,
                                    period_count,
                                    eval_seed_base,
                                    authoritative_lambda,
                                );
                                let score = eval.mean_payoff();
                                let n = eval.games;
                                let wilson = wilson_lower_bound(score, n, 1.96);
                                let branch_best = eval_is_better(
                                    &eval,
                                    increment_best.as_ref().map(|result| &result.eval),
                                );
                                let best_score = best_eval
                                    .as_ref()
                                    .map(EvalStats::mean_payoff)
                                    .unwrap_or(score);
                                println!(
                "increment {inc:>2} cand={candidate_index} lr={learning_rate:.4} lambda={authoritative_lambda:.2} reward_profile={} variant_mode={} train_side={}: games={cumulative:>3}  eval_mean={score:.3}  wilson={wilson:.3}  \
                 global_best={best_score:.3}  w/d/l={}/{}/{}  home={:.3}({}/{}/{}) away={:.3}({}/{}/{})  gd={} ({:.2}/g) gf={} ga={} shots={}:{} sot={}:{} sap={} fwd_margin={} net_fwd_margin={}  {}",
                reward_profile.label,
                seed_variant_mode.label(),
                training_side_mode.label(),
                eval.wins,
                eval.draws,
                eval.losses,
                eval.home_mean_payoff(),
                eval.home_wins,
                eval.home_draws,
                eval.home_losses,
                eval.away_mean_payoff(),
                eval.away_wins,
                eval.away_draws,
                eval.away_losses,
                eval.goal_diff,
                eval.mean_goal_diff(),
                eval.goals_for,
                eval.goals_against,
                eval.shots_for,
                eval.shots_against,
                eval.shots_on_target_for,
                eval.shots_on_target_against,
                eval.shots_after_pass_for,
                eval.forward_pass_margin,
                eval.net_forward_pass_margin,
                if branch_best {
                    "<= BRANCH BEST"
                } else {
                    "(branch discarded)"
                }
            );
                                println!(
                "    side_detail home_gd={} gf={} ga={} shots={}:{} sot={}:{} sap={} fwd={} net_fwd={} | away_gd={} gf={} ga={} shots={}:{} sot={}:{} sap={} fwd={} net_fwd={}",
                eval.home_goal_diff,
                eval.home_goals_for,
                eval.home_goals_against,
                eval.home_shots_for,
                eval.home_shots_against,
                eval.home_shots_on_target_for,
                eval.home_shots_on_target_against,
                eval.home_shots_after_pass_for,
                eval.home_forward_pass_margin,
                eval.home_net_forward_pass_margin,
                eval.away_goal_diff,
                eval.away_goals_for,
                eval.away_goals_against,
                eval.away_shots_for,
                eval.away_shots_against,
                eval.away_shots_on_target_for,
                eval.away_shots_on_target_against,
                eval.away_shots_after_pass_for,
                eval.away_forward_pass_margin,
                eval.away_net_forward_pass_margin,
            );
                                if branch_best {
                                    increment_best = Some(CandidateResult {
                                        learning_rate,
                                        authoritative_lambda,
                                        seed_variant_mode,
                                        training_side_mode,
                                        reward_profile_label: reward_profile.label.clone(),
                                        snapshot: candidate_snapshot,
                                        policies: candidate_policies,
                                        eval,
                                    });
                                }
                                candidate_index = candidate_index.saturating_add(1);
                            }
                        }
                    }
                }
            }

            let Some(increment_best) = increment_best else {
                println!("increment {inc}: no candidate produced a snapshot");
                continue;
            };
            let improved = eval_is_better(&increment_best.eval, best_eval.as_ref());
            if improved {
                best_snapshot = Some(increment_best.snapshot.clone());
                best_policies = Some(increment_best.policies.clone());
                best_eval = Some(increment_best.eval.clone());
                best_at_games = cumulative;
            }
            snapshot = best_snapshot
                .clone()
                .or_else(|| Some(increment_best.snapshot.clone()));
            policies = best_policies
                .clone()
                .unwrap_or_else(|| increment_best.policies.clone());
            let best_score = best_eval
                .as_ref()
                .map(EvalStats::mean_payoff)
                .unwrap_or_else(|| increment_best.eval.mean_payoff());
            println!(
                "increment {inc:>2} selected lr={:.4} lambda={:.2} reward_profile={} variant_mode={} train_side={}: best={best_score:.3}  {}",
                increment_best.learning_rate,
                increment_best.authoritative_lambda,
                increment_best.reward_profile_label,
                increment_best.seed_variant_mode.label(),
                increment_best.training_side_mode.label(),
                if improved {
                    "<= NEW BEST (kept)"
                } else {
                    "(regressed vs best — discarded, ratchet holds)"
                }
            );
        }

        let _ = best_snapshot; // the ratcheted result: the best-seen policy (never regresses)
        let best_score = best_eval
            .as_ref()
            .map(EvalStats::mean_payoff)
            .unwrap_or(-1.0);
        println!(
            "\nRATCHET RESULT: best eval_mean={best_score:.3} at {best_at_games} games \
         (out of {cumulative} trained). The kept policy is the best-seen — monotonic by \
         construction: training noise/divergence can never lower the retained result."
        );
        if best_score > 0.5 {
            println!("=> CLIMBED OUT OF PARITY: best checkpoint beats the analytic engine (>0.5).");
        } else {
            println!(
            "=> still at/under parity (best {best_score:.3} ≤ 0.5): analytic engine not beaten yet; \
             the ratchet at least guarantees we kept the strongest checkpoint."
        );
        }

        if let Some(best) = best_eval.as_ref() {
            aggregate_eval.add_assign(best);
            aggregate_seed_count = aggregate_seed_count.saturating_add(1);
        }
    }

    if aggregate_seed_count > 1 {
        let score = aggregate_eval.mean_payoff();
        let wilson = wilson_lower_bound(score, aggregate_eval.games, 1.96);
        println!(
            "\nMULTI-SEED RATCHET RESULT: seeds={} games={} eval_mean={score:.3} wilson={wilson:.3} \
             w/d/l={}/{}/{} home={:.3}({}/{}/{}) away={:.3}({}/{}/{}) gd={} ({:.2}/g) gf={} ga={} \
             shots={}:{} sot={}:{} sap={} fwd_margin={} net_fwd_margin={}",
            aggregate_seed_count,
            aggregate_eval.games,
            aggregate_eval.wins,
            aggregate_eval.draws,
            aggregate_eval.losses,
            aggregate_eval.home_mean_payoff(),
            aggregate_eval.home_wins,
            aggregate_eval.home_draws,
            aggregate_eval.home_losses,
            aggregate_eval.away_mean_payoff(),
            aggregate_eval.away_wins,
            aggregate_eval.away_draws,
            aggregate_eval.away_losses,
            aggregate_eval.goal_diff,
            aggregate_eval.mean_goal_diff(),
            aggregate_eval.goals_for,
            aggregate_eval.goals_against,
            aggregate_eval.shots_for,
            aggregate_eval.shots_against,
            aggregate_eval.shots_on_target_for,
            aggregate_eval.shots_on_target_against,
            aggregate_eval.shots_after_pass_for,
            aggregate_eval.forward_pass_margin,
            aggregate_eval.net_forward_pass_margin,
        );
        println!(
            "MULTI-SEED SIDE DETAIL: home_gd={} gf={} ga={} shots={}:{} sot={}:{} sap={} fwd={} net_fwd={} | away_gd={} gf={} ga={} shots={}:{} sot={}:{} sap={} fwd={} net_fwd={}",
            aggregate_eval.home_goal_diff,
            aggregate_eval.home_goals_for,
            aggregate_eval.home_goals_against,
            aggregate_eval.home_shots_for,
            aggregate_eval.home_shots_against,
            aggregate_eval.home_shots_on_target_for,
            aggregate_eval.home_shots_on_target_against,
            aggregate_eval.home_shots_after_pass_for,
            aggregate_eval.home_forward_pass_margin,
            aggregate_eval.home_net_forward_pass_margin,
            aggregate_eval.away_goal_diff,
            aggregate_eval.away_goals_for,
            aggregate_eval.away_goals_against,
            aggregate_eval.away_shots_for,
            aggregate_eval.away_shots_against,
            aggregate_eval.away_shots_on_target_for,
            aggregate_eval.away_shots_on_target_against,
            aggregate_eval.away_shots_after_pass_for,
            aggregate_eval.away_forward_pass_margin,
            aggregate_eval.away_net_forward_pass_margin,
        );
        if wilson > 0.5 {
            println!("=> MULTI-SEED STATISTICAL CLIMB: Wilson lower bound is above parity.");
        } else {
            println!("=> multi-seed climb is not yet statistically clear above parity.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_eval_pairs_home_and_away_on_the_same_scenario_seed() {
        let base = 0xd31a_0001;
        assert_eq!(eval_scenario_seed(base, 0), eval_scenario_seed(base, 1));
        assert_eq!(eval_scenario_seed(base, 2), eval_scenario_seed(base, 3));
        assert_eq!(eval_scenario_seed(base, 2), base.wrapping_add(1));
    }

    fn eval(wins: u32, draws: u32, losses: u32, goal_diff: i32, net_forward: i32) -> EvalStats {
        let games = wins + draws + losses;
        EvalStats {
            payoff_sum: wins as f64 + 0.5 * draws as f64,
            games,
            wins,
            draws,
            losses,
            away_payoff_sum: 0.5,
            away_games: 1,
            goal_diff,
            goals_against: losses,
            net_forward_pass_margin: net_forward,
            ..EvalStats::default()
        }
    }

    #[test]
    fn ratchet_eval_prefers_higher_payoff_before_tiebreakers() {
        let incumbent = eval(1, 6, 1, 8, 20);
        let candidate = eval(2, 6, 0, 2, 0);

        assert!(eval_is_better(&candidate, Some(&incumbent)));
        assert!(!eval_is_better(&incumbent, Some(&candidate)));
    }

    #[test]
    fn ratchet_eval_uses_stability_and_net_forward_tiebreakers() {
        let incumbent = eval(2, 6, 0, 2, -2);
        let candidate = eval(2, 6, 0, 2, 0);

        assert!(eval_is_better(&candidate, Some(&incumbent)));
        assert!(!eval_is_better(&incumbent, Some(&candidate)));
    }

    #[test]
    fn ratchet_eval_breaks_payoff_ties_toward_climb_potential() {
        let drawish = EvalStats {
            payoff_sum: 4.5,
            games: 8,
            wins: 2,
            draws: 5,
            losses: 1,
            home_payoff_sum: 3.0,
            home_games: 4,
            away_payoff_sum: 1.5,
            away_games: 4,
            shots_on_target_for: 8,
            shots_on_target_against: 5,
            shots_after_pass_for: 3,
            net_forward_pass_margin: -4,
            ..EvalStats::default()
        };
        let decisive = EvalStats {
            payoff_sum: 4.5,
            games: 8,
            wins: 3,
            draws: 3,
            losses: 2,
            home_payoff_sum: 3.5,
            home_games: 4,
            away_payoff_sum: 1.0,
            away_games: 4,
            goal_diff: 1,
            goals_for: 3,
            goals_against: 2,
            shots_on_target_for: 7,
            shots_on_target_against: 2,
            shots_after_pass_for: 4,
            net_forward_pass_margin: 9,
            ..EvalStats::default()
        };

        assert!(eval_is_better(&decisive, Some(&drawish)));
        assert!(!eval_is_better(&drawish, Some(&decisive)));
    }

    #[test]
    fn ratchet_eval_prefers_balanced_sides_before_goal_diff_tiebreaker() {
        let incumbent = EvalStats {
            payoff_sum: 5.0,
            games: 8,
            wins: 2,
            draws: 6,
            losses: 0,
            home_payoff_sum: 4.0,
            home_games: 4,
            away_payoff_sum: 1.0,
            away_games: 4,
            goal_diff: 6,
            ..EvalStats::default()
        };
        let candidate = EvalStats {
            payoff_sum: 5.0,
            games: 8,
            wins: 2,
            draws: 6,
            losses: 0,
            home_payoff_sum: 2.5,
            home_games: 4,
            away_payoff_sum: 2.5,
            away_games: 4,
            goal_diff: 1,
            ..EvalStats::default()
        };

        assert!(eval_is_better(&candidate, Some(&incumbent)));
        assert!(!eval_is_better(&incumbent, Some(&candidate)));
    }

    #[test]
    fn reward_profile_dedup_uses_weights_not_labels() {
        let base = RewardProfile {
            label: "base".to_string(),
            forward_pass: 6.25,
            pass_chain: 4.25,
            pass_turnover_penalty: 5.0,
            quick_forward_release: 1.25,
            shot_shaping: 0.85,
            shot_commitment: 0.0,
        };
        let mut profiles = vec![base.clone()];

        push_unique_reward_profile(
            &mut profiles,
            RewardProfile {
                label: "same-values".to_string(),
                ..base.clone()
            },
        );
        assert_eq!(profiles.len(), 1);

        push_unique_reward_profile(
            &mut profiles,
            base.scaled("nearby", 1.02, 1.0, 1.0, 1.0, 1.0, 0.01),
        );
        assert_eq!(profiles.len(), 2);
    }

    #[test]
    fn reward_profile_search_sequence_is_low_discrepancy_and_stable() {
        assert_eq!(halton_unit(1, 2), 0.5);
        assert_eq!(halton_unit(2, 2), 0.25);
        assert_eq!(halton_unit(3, 2), 0.75);

        let first = reward_profile_search_multiplier(3, 2, 0.20);
        let repeated = reward_profile_search_multiplier(3, 2, 0.20);
        assert!((first - repeated).abs() <= f64::EPSILON);
        assert!((0.80..=1.20).contains(&first));
    }
}
