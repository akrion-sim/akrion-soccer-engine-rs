//! Postgres persistence for soccer self-play learning.
//!
//! The canonical table contract lives in `remote/libs/pg-defs/schema/schema.sql`.
//! This module is a small Rust adapter over that contract for queue runners.

use native_tls::{Certificate, TlsConnector};
use postgres::types::ToSql;
use postgres::Client;
use postgres_native_tls::MakeTlsConnector;
use serde_json::{json, Value};
use std::fmt::Write as _;
use uuid::Uuid;

use crate::des::general::soccer::{
    MatchConfig, PlayerRole, SoccerConfigComparison, SoccerConfigMomentInsert,
    SoccerMomentEmbeddingInsert, SoccerNeuralNetworkSnapshot, SoccerPassLearningMetrics,
    SoccerPassOutcomeSample, SoccerQEntry, SoccerQPolicy, SoccerQPolicyOptions, SoccerQStateKey,
    SoccerQTargetEntry, SoccerSetPlayTrainingArtifact, SoccerTacticalLearningWeights,
    RECEIVER_DESCRIPTOR_UNSPECIFIED,
    SoccerTeamQPolicies, Team, CONFIG_FEATURE_DIM, CONFIG_FEATURE_DIM_V1,
    SOCCER_MOMENT_EMBEDDING_DIM, SOCCER_PASS_COMPLETION_FEATURE_DIM,
    SOCCER_PASS_COMPLETION_FEATURE_DIM_V1,
};
use crate::des::general::tournament::{
    MatchReport, SoccerTeamGenome, TournamentFormat, TournamentTeam,
};
use crate::des::soccer_learning::{
    soccer_learning_from_micros, soccer_learning_to_micros, soccer_team_label,
    soccer_team_q_policies_fingerprint, SoccerLearningCompletedGame, SoccerLearningPolicyDelta,
    SoccerLearningPolicyDeltaEntry, SoccerLearningPolicyEntryKind, SOCCER_POLICY_STATUS_ACTIVE,
    SOCCER_POLICY_STATUS_ARCHIVED,
};
use std::collections::HashMap;

/// One elite finisher from the breeding pool: its final neural snapshot plus its
/// tactical genome (`None` for rows persisted before genomes shipped).
#[derive(Clone, Debug)]
pub struct TournamentElite {
    pub neural: SoccerNeuralNetworkSnapshot,
    pub genome: Option<SoccerTeamGenome>,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningPgPolicyVersion {
    pub id: String,
    pub generation: i32,
    pub updated_at_micros: i64,
    pub policy_fingerprint: Option<u64>,
    pub policy_entry_count: Option<usize>,
    pub policies: SoccerTeamQPolicies,
    pub neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub tactical_learning: Option<SoccerTacticalLearningWeights>,
    pub search_metadata: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningPgPolicyMetadata {
    pub id: String,
    pub generation: i32,
    pub fitness_micros: i64,
    pub updated_at_micros: i64,
    pub policy_fingerprint: Option<u64>,
    pub policy_entry_count: Option<usize>,
    pub neural_network: Option<SoccerNeuralNetworkSnapshot>,
    pub tactical_learning: Option<SoccerTacticalLearningWeights>,
    pub search_metadata: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct SoccerLearningPgPolicyPromotionBaseline {
    pub policy_version_id: String,
    pub generation: i32,
    pub sample_games: usize,
    pub mean_match_fitness: f64,
    pub best_match_fitness: f64,
    pub mean_play_quality: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SoccerLearningPgCompletedRunInsert<'a> {
    pub base_policy_version_id: Option<&'a str>,
    pub output_policy_version_id: Option<&'a str>,
    pub game: &'a SoccerLearningCompletedGame,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SoccerLearningPgCompletedRunRetentionPrune {
    pub deleted_runs: u64,
    pub deleted_delta_rows: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SoccerLearningPgMomentVectorRetentionPrune {
    pub soft_deleted_moment_embeddings: u64,
    pub soft_deleted_config_moments: u64,
}

const POSTGRES_MAX_QUERY_PARAMETERS: usize = 65_535;
const SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT: usize = 22;
const SOCCER_RUN_DELTA_PARAMETER_COUNT: usize = 17;
const SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT: usize = 9;
const SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT: usize = 14;

const SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE: usize = 1024;
const SOCCER_RUN_DELTA_INSERT_BATCH_SIZE: usize = 1024;
const SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE: usize = 512;
const SOCCER_POLICY_RETENTION_BRANCH_TIP: &str = "branch_tip";
const SOCCER_POLICY_INLINE_PRUNE_ENV: &str = "SOCCER_PG_INLINE_POLICY_PRUNE";
const SOCCER_POLICY_RETAIN_ARCHIVED_ENTRIES_ENV: &str = "SOCCER_PG_RETAIN_ARCHIVED_POLICY_ENTRIES";
const SOCCER_POLICY_VERSION_FULL_ENTRIES_ENV: &str = "SOCCER_PG_POLICY_VERSION_FULL_ENTRIES";
const SOCCER_PG_PERSIST_RUN_DELTAS_ENV: &str = "SOCCER_PG_PERSIST_RUN_DELTAS";
const SOCCER_PG_RESUME_MAX_POLICY_ENTRIES_ENV: &str = "SOCCER_PG_RESUME_MAX_POLICY_ENTRIES";
/// Rows deleted per statement when cleaning up a superseded generation's entries.
/// Inline pruning is opt-in via [`SOCCER_POLICY_INLINE_PRUNE_ENV`]; live learners defer this
/// heavyweight maintenance so policy promotion cannot stall on a large delete.
const SOCCER_POLICY_PRUNE_DELETE_BATCH_SIZE: i64 = 50_000;
pub const SOCCER_MOMENT_VECTOR_SOFT_DELETE_AFTER_DAYS: i64 = 10;

const SOCCER_PG_CONNECT_DEFAULT_MAX_ATTEMPTS: u32 = 5;
const SOCCER_PG_CONNECT_BASE_BACKOFF_MILLIS: u64 = 200;
const SOCCER_PG_CONNECT_MAX_BACKOFF_MILLIS: u64 = 5_000;

// Conflict targets mirror des_soccer_learning_policy_entries_key_uq and
// des_soccer_learning_run_deltas_key_uq. DO NOTHING (not DO UPDATE) is required
// so that duplicate keys appearing within a single multi-row INSERT — e.g. two
// distinct states that collide on the 64-bit state_hash, or an idempotent retry
// re-inserting the same parent's rows — are skipped instead of aborting the
// whole transaction. DO UPDATE would raise "cannot affect row a second time".
/// Page size for streaming a policy version's entries via keyset pagination. Bounds the
/// per-fetch result buffered in client memory (the full decoded policy must still be held —
/// it is the resume state — but the raw result set is no longer materialised all at once).
const SOCCER_POLICY_ENTRY_PAGE_SIZE: i64 = 4096;
const SOCCER_POLICY_ENTRY_ON_CONFLICT_CLAUSE: &str =
    " on conflict (policy_version_id, team, entry_kind, state_hash, action, \
target_fine_cell_id, target_tactical_cell_id, target_macro_cell_id, target_root_cell_id, \
receiver_descriptor) \
do nothing";
const SOCCER_RUN_DELTA_ON_CONFLICT_CLAUSE: &str =
    " on conflict (run_id, team, entry_kind, state_hash, action, \
target_fine_cell_id, target_tactical_cell_id, target_macro_cell_id, target_root_cell_id, \
receiver_descriptor) \
do nothing";
const SOCCER_MOMENT_EMBEDDING_SEARCH_SQL: &str = "\
select team, action, reward_micros, value_micros, tick, \
(embedding <=> $1::text::vector) as distance \
from des_soccer_moment_embeddings \
where ($2::text is null or team = $2) \
  and deleted_at is null \
order by embedding <=> $1::text::vector \
limit $3";
const SOCCER_CONFIG_MOMENT_SEARCH_SQL: &str = "\
select team, role, action, reward_micros, nstep_return_micros, value_micros, \
tick, features, (embedding <=> $1::text::vector) as distance \
from des_soccer_config_moments \
where ($2::text is null or team = $2) \
  and deleted_at is null \
order by embedding <=> $1::text::vector \
limit $3";
/// As [`SOCCER_CONFIG_MOMENT_SEARCH_SQL`] but over the **assigned-position**
/// corpus (slot-aligned ordering). Back-compat rows without the second corpus
/// (null `embedding_assigned`) are skipped so the cosine geometry stays defined.
const SOCCER_CONFIG_MOMENT_SEARCH_ASSIGNED_SQL: &str = "\
select team, role, action, reward_micros, nstep_return_micros, value_micros, \
tick, features_assigned, (embedding_assigned <=> $1::text::vector) as distance \
from des_soccer_config_moments \
where ($2::text is null or team = $2) \
  and deleted_at is null \
  and embedding_assigned is not null \
  and features_assigned is not null \
order by embedding_assigned <=> $1::text::vector \
limit $3";
const SOCCER_MOMENT_EMBEDDING_SOFT_DELETE_SQL: &str = "\
update des_soccer_moment_embeddings \
set deleted_at = now() \
where deleted_at is null \
  and created_at < now() - ($2::bigint * interval '1 day') \
  and ($1::text is null or experiment_id = $1::text::uuid)";
const SOCCER_CONFIG_MOMENT_SOFT_DELETE_SQL: &str = "\
update des_soccer_config_moments \
set deleted_at = now() \
where deleted_at is null \
  and created_at < now() - ($2::bigint * interval '1 day') \
  and ($1::text is null or experiment_id = $1::text::uuid)";

const _: () = {
    assert!(
        SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
    assert!(
        SOCCER_RUN_DELTA_INSERT_BATCH_SIZE * SOCCER_RUN_DELTA_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
    assert!(
        SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
    assert!(
        SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT
            <= POSTGRES_MAX_QUERY_PARAMETERS
    );
};

fn postgres_insert_sql_buffer(prefix: &str, rows: usize, parameters_per_row: usize) -> String {
    let estimated_tuple_bytes = parameters_per_row.saturating_mul(8).saturating_add(24);
    let mut sql = String::with_capacity(
        prefix
            .len()
            .saturating_add(rows.saturating_mul(estimated_tuple_bytes)),
    );
    sql.push_str(prefix);
    sql
}

fn soccer_policy_source_uses_evolutionary_search(source_kind: &str) -> bool {
    soccer_policy_search_label_uses_evolutionary_search(source_kind)
}

fn soccer_policy_search_label_uses_evolutionary_search(label: &str) -> bool {
    let normalized = label
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    normalized.contains("evolution")
        || normalized.contains("genetic")
        || normalized.contains("gp")
        || normalized.contains("mutation")
        || normalized.contains("mutate")
        || normalized.contains("symbolicregression")
        || normalized.contains("programtree")
}

fn soccer_policy_search_metadata_uses_evolutionary_search(metadata: &Value) -> bool {
    match metadata {
        Value::Object(entries) => entries.iter().any(|(key, value)| {
            let key = key.to_ascii_lowercase();
            let key_names_search_label = matches!(
                key.as_str(),
                "algorithm"
                    | "algorithmfamily"
                    | "kind"
                    | "searchalgorithm"
                    | "searchfamily"
                    | "searchkind"
                    | "searchmode"
                    | "mode"
                    | "family"
                    | "sourcekind"
            );
            if key_names_search_label {
                match value {
                    Value::String(label) => {
                        soccer_policy_search_label_uses_evolutionary_search(label)
                    }
                    Value::Array(_) | Value::Object(_) => {
                        soccer_policy_search_metadata_uses_evolutionary_search(value)
                    }
                    _ => false,
                }
            } else {
                matches!(value, Value::Array(_) | Value::Object(_))
                    && soccer_policy_search_metadata_uses_evolutionary_search(value)
            }
        }),
        Value::Array(values) => values
            .iter()
            .any(soccer_policy_search_metadata_uses_evolutionary_search),
        _ => false,
    }
}

fn soccer_team_from_label(label: &str) -> Result<Team, String> {
    match label {
        "home" => Ok(Team::Home),
        "away" => Ok(Team::Away),
        other => Err(format!("unknown soccer team label {other:?}")),
    }
}

fn soccer_policy_entry_kind_from_label(
    label: &str,
) -> Result<SoccerLearningPolicyEntryKind, String> {
    match label {
        "action" => Ok(SoccerLearningPolicyEntryKind::Action),
        "target" => Ok(SoccerLearningPolicyEntryKind::Target),
        other => Err(format!("unknown soccer policy entry kind {other:?}")),
    }
}

fn soccer_policy_version_metrics(
    source_kind: &str,
    fitness: f64,
    neural_network: Option<&SoccerNeuralNetworkSnapshot>,
    tactical_learning: Option<&SoccerTacticalLearningWeights>,
    search_metadata: Option<&Value>,
) -> Result<Value, String> {
    let evolutionary_search = soccer_policy_source_uses_evolutionary_search(source_kind)
        || search_metadata.is_some_and(soccer_policy_search_metadata_uses_evolutionary_search);
    let mut metrics = json!({
        "fitness": fitness,
        "learningComponents": {
            "mdpPolicyEntries": true,
            "pomdpStateKeyFeatures": true,
            "mpcFormationLpWeights": tactical_learning.is_some(),
            "lpAlignmentWeight": tactical_learning
                .map(|weights| weights.formation_lp_alignment_weight)
                .unwrap_or(0.0),
            "neuralNetworkSnapshot": neural_network.is_some()
        },
        "learningProvenance": {
            "sourceKind": source_kind,
            "evolutionarySearch": evolutionary_search,
            "geneticProgrammingTacticalSearch": evolutionary_search && tactical_learning.is_some(),
            "tacticalLearningWeightsPersisted": tactical_learning.is_some(),
            "neuralNetworkSnapshotPersisted": neural_network.is_some()
        }
    });
    if let Some(neural_network) = neural_network {
        validate_soccer_neural_network_snapshot_for_pg(neural_network)?;
        metrics["neuralNetwork"] = serde_json::to_value(neural_network)
            .map_err(|err| format!("serialize soccer neural network snapshot: {err}"))?;
    }
    if let Some(tactical_learning) = tactical_learning {
        validate_soccer_tactical_learning_weights_for_pg(tactical_learning)?;
        metrics["tacticalLearning"] = serde_json::to_value(tactical_learning)
            .map_err(|err| format!("serialize soccer tactical learning weights: {err}"))?;
    }
    if let Some(search_metadata) = search_metadata {
        if !search_metadata.is_object() {
            return Err("soccer policy search metadata must be a JSON object".to_string());
        }
        metrics["learningProvenance"]["searchParameters"] = search_metadata.clone();
    }
    Ok(metrics)
}

fn soccer_policy_entry_count(policies: &SoccerTeamQPolicies) -> usize {
    policies.home.entries().len()
        + policies.away.entries().len()
        + policies.home.target_entries().len()
        + policies.away.target_entries().len()
}

fn stamp_soccer_policy_weight_metrics(metrics: &mut Value, policies: &SoccerTeamQPolicies) {
    metrics["learningComponents"]["policyFingerprint"] =
        json!(soccer_team_q_policies_fingerprint(policies));
    metrics["learningComponents"]["policyEntryCount"] = json!(soccer_policy_entry_count(policies));
    metrics["learningProvenance"]["policyWeightsPersisted"] = json!(true);
}

fn soccer_policy_version_metrics_with_retention(
    mut metrics: Value,
    policy_version_id: &str,
    branch_key: &str,
    retention_kind: &str,
    full_entries_retained: bool,
) -> Value {
    metrics["postgresRetention"] = json!({
        "policyVersionId": policy_version_id,
        "branchKey": branch_key,
        "retentionKind": retention_kind,
        "fullEntriesRetained": full_entries_retained,
    });
    metrics
}

fn soccer_policy_version_neural_network_from_metrics(
    metrics: &Value,
) -> Result<Option<SoccerNeuralNetworkSnapshot>, String> {
    let Some(neural_network) = metrics.get("neuralNetwork") else {
        return Ok(None);
    };
    let snapshot = serde_json::from_value::<SoccerNeuralNetworkSnapshot>(neural_network.clone())
        .map_err(|err| format!("decode soccer neural network snapshot: {err}"))?;
    validate_soccer_neural_network_snapshot_for_pg(&snapshot)?;
    Ok(Some(snapshot))
}

fn soccer_policy_version_tactical_learning_from_values(
    config: &Value,
    metrics: &Value,
) -> Result<Option<SoccerTacticalLearningWeights>, String> {
    let tactical_learning = metrics
        .get("tacticalLearning")
        .or_else(|| config.get("tacticalLearning"));
    let Some(tactical_learning) = tactical_learning else {
        return Ok(None);
    };
    let weights =
        serde_json::from_value::<SoccerTacticalLearningWeights>(tactical_learning.clone())
            .map_err(|err| format!("decode soccer tactical learning weights: {err}"))?;
    validate_soccer_tactical_learning_weights_for_pg(&weights)?;
    Ok(Some(weights))
}

fn soccer_policy_version_search_metadata_from_metrics(
    metrics: &Value,
) -> Result<Option<Value>, String> {
    let Some(search_metadata) = metrics
        .get("learningProvenance")
        .and_then(|provenance| provenance.get("searchParameters"))
    else {
        return Ok(None);
    };
    if !search_metadata.is_object() {
        return Err("soccer policy search metadata must be a JSON object".to_string());
    }
    Ok(Some(search_metadata.clone()))
}

fn soccer_policy_version_policy_fingerprint_from_metrics(metrics: &Value) -> Option<u64> {
    metrics
        .get("learningComponents")
        .and_then(|components| components.get("policyFingerprint"))
        .or_else(|| metrics.get("policyFingerprint"))
        .and_then(Value::as_u64)
}

fn soccer_policy_version_policy_entry_count_from_metrics(metrics: &Value) -> Option<usize> {
    metrics
        .get("learningComponents")
        .and_then(|components| components.get("policyEntryCount"))
        .or_else(|| metrics.get("policyEntryCount"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn soccer_policy_version_retains_full_entries(status: &str) -> bool {
    soccer_policy_version_retains_full_entries_with_override(
        status,
        soccer_learning_pg_env_flag(SOCCER_POLICY_RETAIN_ARCHIVED_ENTRIES_ENV),
    )
}

fn soccer_policy_version_status_after_promotion_sample_floor<'a>(
    requested_status: &'a str,
    metrics: &Value,
) -> &'a str {
    if requested_status != SOCCER_POLICY_STATUS_ACTIVE {
        return requested_status;
    }
    let Some(gate) = metrics
        .get("learningProvenance")
        .and_then(|provenance| provenance.get("searchParameters"))
        .and_then(|search_parameters| search_parameters.get("promotion"))
        .and_then(|promotion| promotion.get("gate"))
    else {
        return requested_status;
    };
    let sample_games = gate.get("sampleGames").and_then(Value::as_u64);
    let configured_min_sample_games = soccer_neural_snapshot_min_sample_games().max(0) as u64;
    let min_sample_games = gate
        .get("minSampleGames")
        .and_then(Value::as_u64)
        .unwrap_or(configured_min_sample_games)
        .max(configured_min_sample_games);
    if sample_games.is_some_and(|sample_games| sample_games >= min_sample_games) {
        requested_status
    } else {
        SOCCER_POLICY_STATUS_ARCHIVED
    }
}

fn soccer_policy_version_retains_full_entries_with_override(
    status: &str,
    retain_archived_entries: bool,
) -> bool {
    if retain_archived_entries {
        return !matches!(status, "rejected");
    }
    !matches!(status, "archived" | "rejected")
}

fn validate_soccer_tactical_learning_weights_for_pg(
    weights: &SoccerTacticalLearningWeights,
) -> Result<(), String> {
    for (name, value) in [
        (
            "attackSpacingDeltaWeight",
            weights.attack_spacing_delta_weight,
        ),
        (
            "attackSpacingScoreWeight",
            weights.attack_spacing_score_weight,
        ),
        ("attackWidthDeltaWeight", weights.attack_width_delta_weight),
        ("attackWidthScoreWeight", weights.attack_width_score_weight),
        ("attackFlankLaneWeight", weights.attack_flank_lane_weight),
        (
            "defenseSpacingDeltaWeight",
            weights.defense_spacing_delta_weight,
        ),
        (
            "defenseSpacingScoreWeight",
            weights.defense_spacing_score_weight,
        ),
        (
            "defenseContractDeltaWeight",
            weights.defense_contract_delta_weight,
        ),
        (
            "defenseCompactnessScoreWeight",
            weights.defense_compactness_score_weight,
        ),
        (
            "defenseBallDepthScoreWeight",
            weights.defense_ball_depth_score_weight,
        ),
        (
            "defenseEndlineSoftPenaltyWeight",
            weights.defense_endline_soft_penalty_weight,
        ),
        (
            "defenseEndlineHardPenaltyWeight",
            weights.defense_endline_hard_penalty_weight,
        ),
        (
            "defenderMidfielderPressWeight",
            weights.defender_midfielder_press_weight,
        ),
        ("midfielderPressWeight", weights.midfielder_press_weight),
        (
            "formationLpAlignmentWeight",
            weights.formation_lp_alignment_weight,
        ),
    ] {
        if !value.is_finite() {
            return Err(format!(
                "{name} must be finite in soccer tactical learning weights"
            ));
        }
    }
    Ok(())
}

fn validate_soccer_neural_network_snapshot_for_pg(
    snapshot: &SoccerNeuralNetworkSnapshot,
) -> Result<(), String> {
    if snapshot.input_dim == 0 {
        return Err("soccer neural network snapshot input_dim must be positive".to_string());
    }
    if snapshot.output_dim == 0 {
        return Err("soccer neural network snapshot output_dim must be positive".to_string());
    }
    if snapshot.layers.is_empty() {
        return Err("soccer neural network snapshot must contain layers".to_string());
    }
    if !snapshot.l2_norm.is_finite() || snapshot.l2_norm < 0.0 {
        return Err(
            "soccer neural network snapshot l2_norm must be finite and non-negative".to_string(),
        );
    }
    let mut expected_input_dim = snapshot.input_dim;
    let mut parameter_count = 0usize;
    for (layer_idx, layer) in snapshot.layers.iter().enumerate() {
        let activation = layer.activation.to_ascii_lowercase();
        if !matches!(activation.as_str(), "linear" | "sigmoid" | "tanh" | "relu") {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has unsupported activation {:?}",
                layer.activation
            ));
        }
        let output_dim = layer.biases.len();
        if output_dim == 0 {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has no biases"
            ));
        }
        if layer.weights.len() != output_dim {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has {} weight rows for {} biases",
                layer.weights.len(),
                output_dim
            ));
        }
        for (row_idx, row) in layer.weights.iter().enumerate() {
            if row.len() != expected_input_dim {
                return Err(format!(
                    "soccer neural network snapshot layer {layer_idx} row {row_idx} has input dim {}, expected {}",
                    row.len(),
                    expected_input_dim
                ));
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "soccer neural network snapshot layer {layer_idx} row {row_idx} has non-finite weights"
                ));
            }
            parameter_count = parameter_count.saturating_add(row.len());
        }
        if layer.biases.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "soccer neural network snapshot layer {layer_idx} has non-finite biases"
            ));
        }
        parameter_count = parameter_count.saturating_add(layer.biases.len());
        expected_input_dim = output_dim;
    }
    if expected_input_dim != snapshot.output_dim {
        return Err(format!(
            "soccer neural network snapshot final output dim {expected_input_dim} does not match declared {}",
            snapshot.output_dim
        ));
    }
    if snapshot.parameter_count != 0 && snapshot.parameter_count != parameter_count {
        return Err(format!(
            "soccer neural network snapshot parameter_count {} does not match decoded {}",
            snapshot.parameter_count, parameter_count
        ));
    }
    Ok(())
}

pub struct SoccerLearningPgStore {
    client: Client,
    /// Retained so a connection dropped mid-session (e.g. RDS failover, idle socket reaped)
    /// can be rebuilt — `connect_with_retry` originally covered only the first connect.
    database_url: String,
    policy_retention_schema_ready: bool,
}

impl SoccerLearningPgStore {
    pub fn connect(database_url: &str) -> Result<Self, String> {
        Self::connect_with_retry(database_url, soccer_learning_pg_connect_max_attempts())
    }

    /// Rebuild the client (with the same retry/backoff as the initial connect) after the
    /// session has dropped. Re-applies the session timeouts via `connect_with_retry`.
    fn reconnect(&mut self) -> Result<(), String> {
        let rebuilt = Self::connect_with_retry(
            &self.database_url,
            soccer_learning_pg_connect_max_attempts(),
        )?;
        self.client = rebuilt.client;
        self.policy_retention_schema_ready = rebuilt.policy_retention_schema_ready;
        Ok(())
    }

    /// Reconnect if the connection has been closed since the last operation. Called at the
    /// top of the store's public entry points so a long-lived queue/server loop self-heals
    /// across a failover instead of erroring on every subsequent call until restart.
    fn ensure_connected(&mut self) -> Result<(), String> {
        if self.client.is_closed() {
            eprintln!(
                "soccer-learning-pg: connection closed since last use; reconnecting before next operation"
            );
            self.reconnect()?;
        }
        Ok(())
    }

    fn ensure_policy_retention_schema_ready(&mut self) -> Result<(), String> {
        if self.policy_retention_schema_ready {
            return Ok(());
        }
        if self.policy_retention_columns_present()? {
            self.policy_retention_schema_ready = true;
            return Ok(());
        }
        self.with_transient_retry("ensure soccer policy retention schema", |store| {
            let mut tx = store.client.transaction().map_err(|err| {
                format!("begin soccer policy retention schema transaction: {err}")
            })?;
            ensure_soccer_learning_policy_retention_columns(&mut tx)?;
            tx.commit().map_err(|err| {
                format!("commit soccer policy retention schema transaction: {err}")
            })?;
            Ok(())
        })?;
        self.policy_retention_schema_ready = true;
        Ok(())
    }

    fn policy_retention_columns_present(&mut self) -> Result<bool, String> {
        self.with_transient_retry("check soccer policy retention columns", |store| {
            let row = store
                .client
                .query_one(
                    r#"
                    select count(*) = 4
                    from information_schema.columns
                    where table_schema = current_schema()
                      and table_name = 'des_soccer_learning_policy_versions'
                      and column_name in (
                        'branch_key',
                        'retention_kind',
                        'full_entries_retained',
                        'full_entries_pruned_at'
                      )
                    "#,
                    &[],
                )
                .map_err(|err| format!("check soccer policy retention columns: {err}"))?;
            Ok(row.get(0))
        })
    }

    /// Run an **idempotent** database operation, retrying on a transient connection
    /// drop (RDS failover, network blip, server restart) with the same bounded
    /// exponential backoff as the initial connect, reconnecting between attempts.
    ///
    /// SAFETY — `op` is re-run verbatim after a failure, so it MUST be idempotent.
    /// We retry ONLY when the session itself closed (`is_closed()` became true after
    /// the failure); a constraint/logic/bad-input error leaves the connection
    /// usable, so it is returned immediately and never looped. The one subtle case a
    /// connection drop can hide is "the COMMIT succeeded on the server but the ack
    /// was lost when the socket died" — re-running then re-applies the same rows, so
    /// wrapped callers MUST use a stable key + `on conflict do nothing/update` (or a
    /// `where id = …` update) so that re-apply is a no-op, never a duplicate. This is
    /// why the server-id-generating inserts (`insert_completed_run`) are deliberately
    /// NOT wrapped.
    fn with_transient_retry<T>(
        &mut self,
        what: &str,
        mut op: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let max_attempts = soccer_learning_pg_connect_max_attempts();
        let mut attempt = 1u32;
        loop {
            self.ensure_connected()?;
            match op(self) {
                Ok(value) => return Ok(value),
                Err(err) => {
                    // A still-open connection means a real error (constraint, bad
                    // input, statement timeout) — surface it, don't loop. Only a
                    // dropped session is the safe, transient case to retry.
                    if !self.client.is_closed() || attempt >= max_attempts {
                        return Err(err);
                    }
                    let backoff = soccer_learning_pg_retry_backoff(attempt);
                    eprintln!(
                        "soccer-learning-pg: transient connection drop on {what} \
                         (attempt {attempt}/{max_attempts}), reconnecting + retrying in {}ms: {err}",
                        backoff.as_millis()
                    );
                    std::thread::sleep(backoff);
                    // Rebuild now so the next attempt starts on a live session (the
                    // loop-top ensure_connected is a backstop if this also fails).
                    let _ = self.reconnect();
                    attempt += 1;
                }
            }
        }
    }

    /// Try to hold a session-level Postgres advisory lock for a whole tournament run.
    ///
    /// This is intentionally not transaction-scoped: AWS and Hetzner CronJobs can
    /// both fire at 2am, but only the session that owns this lock should proceed
    /// to run matches. The lock releases automatically if the process exits or the
    /// database connection drops.
    pub fn try_acquire_tournament_run_lock(&mut self, key: &str) -> Result<bool, String> {
        self.ensure_connected()?;
        let row = self
            .client
            .query_one(
                "select pg_try_advisory_lock(hashtext('des-soccer-tournament-run'), hashtext($1))",
                &[&key],
            )
            .map_err(|err| format!("acquire tournament run advisory lock: {err}"))?;
        Ok(row.get(0))
    }

    pub fn release_tournament_run_lock(&mut self, key: &str) -> Result<bool, String> {
        self.ensure_connected()?;
        let row = self
            .client
            .query_one(
                "select pg_advisory_unlock(hashtext('des-soccer-tournament-run'), hashtext($1))",
                &[&key],
            )
            .map_err(|err| format!("release tournament run advisory lock: {err}"))?;
        Ok(row.get(0))
    }

    fn build_tls_connector(database_url: &str) -> Result<MakeTlsConnector, String> {
        let mut tls_builder = TlsConnector::builder();
        if soccer_learning_pg_should_verify_certificates(database_url) {
            // Secure by default: authenticate the server certificate. Operators
            // on minimal images whose system trust store lacks the RDS CA can
            // pin an explicit bundle (e.g. the AWS RDS global root) via
            // SOCCER_PG_SSLROOTCERT / PGSSLROOTCERT instead of disabling checks.
            for certificate in soccer_learning_pg_root_certificates()? {
                tls_builder.add_root_certificate(certificate);
            }
        } else {
            // Explicit operator opt-out only (sslmode=disable/allow/prefer or
            // SOCCER_PG_TLS_INSECURE): encrypt the wire but skip verification.
            eprintln!(
                "soccer-learning-pg: TLS certificate verification disabled via opt-out; \
                 connection is encrypted but unauthenticated"
            );
            tls_builder.danger_accept_invalid_certs(true);
            tls_builder.danger_accept_invalid_hostnames(true);
        }
        let tls = tls_builder
            .build()
            .map_err(|err| format!("build soccer learning postgres tls connector: {err}"))?;
        Ok(MakeTlsConnector::new(tls))
    }

    fn connect_with_retry(database_url: &str, max_attempts: u32) -> Result<Self, String> {
        let max_attempts = max_attempts.max(1);
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let tls = Self::build_tls_connector(database_url)?;
            match Client::connect(database_url, tls) {
                Ok(mut client) => {
                    // Bound server-side work so a slow/stuck query or an abandoned open
                    // transaction can't hang a long-lived queue/server loop indefinitely.
                    let statement_timeout = soccer_learning_pg_statement_timeout();
                    if let Err(err) = client.batch_execute(&format!(
                        "SET statement_timeout = '{statement_timeout}'; \
                         SET idle_in_transaction_session_timeout = '60s'"
                    )) {
                        return Err(format!(
                            "set soccer learning postgres session timeouts: {err}"
                        ));
                    }
                    return Ok(Self {
                        client,
                        database_url: database_url.to_string(),
                        policy_retention_schema_ready: false,
                    });
                }
                Err(err) => {
                    let transient = soccer_learning_pg_error_is_transient(&err);
                    if attempt >= max_attempts || !transient {
                        return Err(format!(
                            "connect soccer learning postgres (attempt {attempt}/{max_attempts}): {err}"
                        ));
                    }
                    let backoff = soccer_learning_pg_retry_backoff(attempt);
                    eprintln!(
                        "soccer-learning-pg: transient connect error on attempt \
                         {attempt}/{max_attempts}, retrying in {}ms: {err}",
                        backoff.as_millis()
                    );
                    std::thread::sleep(backoff);
                }
            }
        }
    }

    pub fn connect_from_env() -> Result<Option<Self>, String> {
        let Some(database_url) = soccer_learning_database_url() else {
            return Ok(None);
        };
        Self::connect(&database_url).map(Some)
    }

    pub fn ensure_experiment(
        &mut self,
        slug: &str,
        display_name: &str,
        config: &MatchConfig,
    ) -> Result<String, String> {
        self.ensure_connected()?;
        if let Some(row) = self
            .client
            .query_opt(
                r#"
                select id::text
                from des_soccer_learning_experiments
                where slug = $1 and is_soft_deleted = false
                limit 1
                "#,
                &[&slug],
            )
            .map_err(|err| format!("select soccer learning experiment: {err}"))?
        {
            return Ok(row.get(0));
        }

        let config_json =
            serde_json::to_value(config).map_err(|err| format!("serialize match config: {err}"))?;
        let row = self
            .client
            .query_one(
                r#"
                insert into des_soccer_learning_experiments
                  (slug, display_name, config, meta_data)
                values
                  ($1, $2, $3, '{}'::jsonb)
                returning id::text
                "#,
                &[&slug, &display_name, &config_json],
            )
            .map_err(|err| format!("insert soccer learning experiment: {err}"))?;
        Ok(row.get(0))
    }

    /// Synchronously load the process-wide tunables overlay from Postgres: the
    /// most recent row of `des_soccer_tunables`, whose `overlay` is a partial JSON
    /// patch applied on top of the compile-time defaults and the environment layer
    /// (see [`crate::des::general::soccer::Tunables`]). The table is created on
    /// demand, so a fresh database simply yields `None` (env + defaults stand).
    /// Blocking; intended to be called once at startup before the engine reads any
    /// tunable.
    pub fn fetch_tunables_overlay(&mut self) -> Result<Option<Value>, String> {
        self.ensure_connected()?;
        self.client
            .batch_execute(
                r#"
                create table if not exists des_soccer_tunables (
                    id bigserial primary key,
                    created_at timestamptz not null default now(),
                    note text,
                    overlay jsonb not null
                )
                "#,
            )
            .map_err(|err| format!("ensure des_soccer_tunables: {err}"))?;
        let row = self
            .client
            .query_opt(
                r#"
                select overlay
                from des_soccer_tunables
                order by id desc
                limit 1
                "#,
                &[],
            )
            .map_err(|err| format!("select soccer tunables overlay: {err}"))?;
        Ok(row.map(|row| row.get::<_, Value>(0)))
    }

    /// Durably store a finished learning tournament: one header row, every match,
    /// and every team's final evolved brain (neural snapshot + record). Returns the
    /// new tournament id.
    ///
    /// A transaction-scoped advisory lock keyed by `(experiment, date)` serializes
    /// concurrent writers so two nightly runs for the same date can't interleave —
    /// the second blocks until the first commits, then writes its own row.
    /// Create a tournament header at START (status='running', champion unknown) and return
    /// its id, so progress can be persisted AS THE BRACKET PLAYS (a crash then leaves a
    /// durable record instead of nothing). Serializes nightly runs for the same
    /// (experiment, date) via an advisory lock held only for the header insert.
    #[allow(clippy::too_many_arguments)]
    pub fn start_tournament(
        &mut self,
        experiment_id: &str,
        tournament_date: &str,
        seed: u32,
        learning_mode: &str,
        format: &TournamentFormat,
        match_total: usize,
    ) -> Result<i64, String> {
        self.ensure_connected()?;
        let format_json = json!({
            "teamCount": format.team_count,
            "groupSize": format.group_size,
            "advancersPerGroup": format.advancers_per_group,
            "doubleRoundRobin": format.double_round_robin,
            "thirdPlaceMatch": format.third_place_match,
        });
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin tournament start tx: {err}"))?;
        ensure_soccer_tournament_tables(&mut tx)?;
        // Serialize concurrent nightly runs for the same (experiment, date).
        tx.execute(
            "select pg_advisory_xact_lock(hashtext($1), hashtext($2))",
            &[&experiment_id, &tournament_date],
        )
        .map_err(|err| format!("acquire tournament advisory lock: {err}"))?;
        let tournament_id: i64 = tx
            .query_one(
                r#"
                insert into des_soccer_tournaments
                  (experiment_id, tournament_date, seed, learning_mode, format,
                   team_count, match_count, status)
                values ($1::text::uuid, $2, $3, $4, $5, $6, $7, 'running')
                returning id
                "#,
                &[
                    &experiment_id,
                    &tournament_date,
                    &i64::from(seed),
                    &learning_mode,
                    &format_json,
                    &(format.team_count as i32),
                    &(match_total as i32),
                ],
            )
            .map_err(|err| format!("insert tournament header: {err}"))?
            .get(0);
        tx.commit()
            .map_err(|err| format!("commit tournament header: {err}"))?;
        Ok(tournament_id)
    }

    /// Persist a batch of just-played matches (idempotent via ON CONFLICT) and refresh
    /// matches_played. `index_offset` is the flat match index of `matches[0]` within the
    /// bracket (the caller passes the count already persisted), so re-running a wave is a
    /// no-op rather than a duplicate.
    pub fn record_tournament_matches(
        &mut self,
        tournament_id: i64,
        matches: &[MatchReport],
        index_offset: usize,
    ) -> Result<(), String> {
        if matches.is_empty() {
            return Ok(());
        }
        // Idempotent (on conflict (tournament_id, match_index) do nothing, with
        // stable keys), so transparently retry across a transient connection drop.
        self.with_transient_retry("record tournament matches", |store| {
            store.record_tournament_matches_inner(tournament_id, matches, index_offset)
        })
    }

    fn record_tournament_matches_inner(
        &mut self,
        tournament_id: i64,
        matches: &[MatchReport],
        index_offset: usize,
    ) -> Result<(), String> {
        self.ensure_connected()?;
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin tournament-matches tx: {err}"))?;
        let stmt = tx
            .prepare(
                r#"
                insert into des_soccer_tournament_matches
                  (tournament_id, match_index, stage, home_team_id, away_team_id,
                   home_goals, away_goals, shootout_winner_team_id,
                   home_training_steps, away_training_steps)
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                on conflict (tournament_id, match_index) do nothing
                "#,
            )
            .map_err(|err| format!("prepare tournament-match insert: {err}"))?;
        for (offset, m) in matches.iter().enumerate() {
            let match_index = (index_offset + offset) as i32;
            let shootout = m.shootout_winner.map(|id| id as i32);
            tx.execute(
                &stmt,
                &[
                    &tournament_id,
                    &match_index,
                    &m.stage.label(),
                    &(m.home_id as i32),
                    &(m.away_id as i32),
                    &(m.home_goals as i32),
                    &(m.away_goals as i32),
                    &shootout,
                    &(m.home_training_steps as i64),
                    &(m.away_training_steps as i64),
                ],
            )
            .map_err(|err| format!("insert tournament match {match_index}: {err}"))?;
        }
        tx.execute(
            r#"
            update des_soccer_tournaments
            set matches_played = (
                  select count(*) from des_soccer_tournament_matches
                  where tournament_id = $1
                ),
                updated_at = now()
            where id = $1
            "#,
            &[&tournament_id],
        )
        .map_err(|err| format!("update tournament matches_played: {err}"))?;
        tx.commit()
            .map_err(|err| format!("commit tournament matches: {err}"))?;
        Ok(())
    }

    /// Checkpoint each team's evolving brain (record + neural snapshot) — upserted so the
    /// row reflects the latest wave. After a crash, the best of these is the salvageable
    /// policy (and `load_latest_tournament_team_brains` warm-starts the next run from them).
    pub fn checkpoint_tournament_team_brains(
        &mut self,
        tournament_id: i64,
        teams: &[TournamentTeam],
    ) -> Result<(), String> {
        if teams.is_empty() {
            return Ok(());
        }
        // Idempotent (on conflict (tournament_id, team_id) do update, stable keys),
        // so retry across a transient drop — this persists the evolved brains, the
        // run's whole learning payload, so resilience here matters most.
        self.with_transient_retry("checkpoint tournament team brains", |store| {
            store.checkpoint_tournament_team_brains_inner(tournament_id, teams)
        })
    }

    fn checkpoint_tournament_team_brains_inner(
        &mut self,
        tournament_id: i64,
        teams: &[TournamentTeam],
    ) -> Result<(), String> {
        self.ensure_connected()?;
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin team-brains tx: {err}"))?;
        let stmt = tx
            .prepare(
                r#"
                insert into des_soccer_tournament_team_brains
                  (tournament_id, team_id, team_name, seed, matches_learned,
                   training_steps, played, wins, draws, losses, goals_for,
                   goals_against, neural_snapshot, genome, updated_at)
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, now())
                on conflict (tournament_id, team_id) do update set
                  team_name = excluded.team_name,
                  matches_learned = excluded.matches_learned,
                  training_steps = excluded.training_steps,
                  played = excluded.played,
                  wins = excluded.wins,
                  draws = excluded.draws,
                  losses = excluded.losses,
                  goals_for = excluded.goals_for,
                  goals_against = excluded.goals_against,
                  neural_snapshot = excluded.neural_snapshot,
                  genome = excluded.genome,
                  updated_at = now()
                "#,
            )
            .map_err(|err| format!("prepare team-brain upsert: {err}"))?;
        for team in teams {
            let neural_json = match &team.brain.neural {
                Some(snapshot) => {
                    validate_soccer_neural_network_snapshot_for_pg(snapshot)?;
                    Some(
                        serde_json::to_value(snapshot)
                            .map_err(|err| format!("serialize team {} brain: {err}", team.id))?,
                    )
                }
                None => None,
            };
            let genome_json = serde_json::to_value(&team.brain.genome)
                .map_err(|err| format!("serialize team {} genome: {err}", team.id))?;
            tx.execute(
                &stmt,
                &[
                    &tournament_id,
                    &(team.id as i32),
                    &team.name,
                    &(team.seed as i64),
                    &(team.brain.matches_learned as i32),
                    &(team.brain.training_steps as i64),
                    &(team.record.played as i32),
                    &(team.record.wins as i32),
                    &(team.record.draws as i32),
                    &(team.record.losses as i32),
                    &(team.record.goals_for as i32),
                    &(team.record.goals_against as i32),
                    &neural_json,
                    &genome_json,
                ],
            )
            .map_err(|err| format!("upsert team {} brain: {err}", team.id))?;
        }
        tx.commit()
            .map_err(|err| format!("commit team brains: {err}"))?;
        Ok(())
    }

    /// Finalize the header: status (completed/failed/aborted) + champion + finish time.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_tournament(
        &mut self,
        tournament_id: i64,
        status: &str,
        champion_team_id: Option<usize>,
        runner_up_team_id: Option<usize>,
        third_place_team_id: Option<usize>,
        wall_time_seconds: f64,
    ) -> Result<(), String> {
        // Idempotent same-value UPDATE keyed by id, so retry across a transient drop.
        self.with_transient_retry("finish tournament", |store| {
            store.finish_tournament_inner(
                tournament_id,
                status,
                champion_team_id,
                runner_up_team_id,
                third_place_team_id,
                wall_time_seconds,
            )
        })
    }

    fn finish_tournament_inner(
        &mut self,
        tournament_id: i64,
        status: &str,
        champion_team_id: Option<usize>,
        runner_up_team_id: Option<usize>,
        third_place_team_id: Option<usize>,
        wall_time_seconds: f64,
    ) -> Result<(), String> {
        self.ensure_connected()?;
        let champion = champion_team_id.map(|id| id as i32);
        let runner_up = runner_up_team_id.map(|id| id as i32);
        let third = third_place_team_id.map(|id| id as i32);
        self.client
            .execute(
                r#"
                update des_soccer_tournaments
                set status = $2, champion_team_id = $3, runner_up_team_id = $4,
                    third_place_team_id = $5, wall_time_seconds = $6,
                    updated_at = now(), finished_at = now()
                where id = $1
                "#,
                &[
                    &tournament_id,
                    &status,
                    &champion,
                    &runner_up,
                    &third,
                    &wall_time_seconds,
                ],
            )
            .map_err(|err| format!("finish tournament: {err}"))?;
        Ok(())
    }

    /// Warm-start brains for the next tournament: each team's final neural snapshot
    /// from this experiment's most recently completed tournament, keyed by team id.
    /// Empty when no prior tournament exists (callers start those teams fresh).
    pub fn load_latest_tournament_team_brains(
        &mut self,
        experiment_id: &str,
    ) -> Result<HashMap<usize, SoccerNeuralNetworkSnapshot>, String> {
        self.ensure_connected()?;
        {
            let mut tx = self
                .client
                .transaction()
                .map_err(|err| format!("begin tournament read schema tx: {err}"))?;
            ensure_soccer_tournament_tables(&mut tx)?;
            tx.commit()
                .map_err(|err| format!("commit tournament read schema: {err}"))?;
        }
        let rows = self
            .client
            .query(
                r#"
                select b.team_id, b.neural_snapshot
                from des_soccer_tournament_team_brains b
                where b.tournament_id = (
                  select id from des_soccer_tournaments
                  where experiment_id = $1::text::uuid
                    and status in ('completed', 'partial')
                  order by created_at desc, id desc
                  limit 1
                )
                and b.neural_snapshot is not null
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("load latest tournament brains: {err}"))?;
        let mut brains = HashMap::with_capacity(rows.len());
        for row in rows {
            let team_id: i32 = row.get(0);
            let snapshot_json: Value = row.get(1);
            let snapshot: SoccerNeuralNetworkSnapshot = serde_json::from_value(snapshot_json)
                .map_err(|err| format!("decode team {team_id} brain: {err}"))?;
            validate_soccer_neural_network_snapshot_for_pg(&snapshot)?;
            brains.insert(team_id.max(0) as usize, snapshot);
        }
        Ok(brains)
    }

    /// The breeding pool for the next generation: the top-`limit` finishers of
    /// this experiment's most recently completed tournament, as an ordered Vec of
    /// their final neural snapshots (best first). Finish is ranked by matches
    /// advanced (a clean proxy for bracket depth — the champion plays the most),
    /// then league points, then goal difference. Empty when no completed
    /// tournament exists (callers fall back to the active policy / a cold field).
    pub fn load_latest_tournament_elite_neural_pool(
        &mut self,
        experiment_id: &str,
        limit: usize,
    ) -> Result<Vec<TournamentElite>, String> {
        self.ensure_connected()?;
        {
            let mut tx = self
                .client
                .transaction()
                .map_err(|err| format!("begin tournament elite read schema tx: {err}"))?;
            ensure_soccer_tournament_tables(&mut tx)?;
            tx.commit()
                .map_err(|err| format!("commit tournament elite read schema: {err}"))?;
        }
        let limit_i64 = limit.max(1) as i64;
        let rows = self
            .client
            .query(
                r#"
                select b.neural_snapshot, b.genome
                from des_soccer_tournament_team_brains b
                where b.tournament_id = (
                  select id from des_soccer_tournaments
                  where experiment_id = $1::text::uuid
                    and status in ('completed', 'partial')
                  order by created_at desc, id desc
                  limit 1
                )
                and b.neural_snapshot is not null
                order by b.played desc,
                         (b.wins * 3 + b.draws) desc,
                         (b.goals_for - b.goals_against) desc,
                         b.goals_for desc,
                         b.team_id asc
                limit $2
                "#,
                &[&experiment_id, &limit_i64],
            )
            .map_err(|err| format!("load tournament elite pool: {err}"))?;
        let mut pool = Vec::with_capacity(rows.len());
        for row in rows {
            let snapshot_json: Value = row.get(0);
            let snapshot: SoccerNeuralNetworkSnapshot = serde_json::from_value(snapshot_json)
                .map_err(|err| format!("decode elite brain: {err}"))?;
            validate_soccer_neural_network_snapshot_for_pg(&snapshot)?;
            // Genome is nullable (rows persisted before the genome shipped have
            // none) — decode best-effort so a missing/old genome never blocks the
            // breeding pool, and sanitize so a corrupt/out-of-range persisted genome
            // can't reach the engine or the GA.
            let genome_json: Option<Value> = row.get(1);
            let genome = genome_json
                .and_then(|value| serde_json::from_value::<SoccerTeamGenome>(value).ok())
                .map(|mut g| {
                    g.sanitize();
                    g
                });
            pool.push(TournamentElite {
                neural: snapshot,
                genome,
            });
        }
        Ok(pool)
    }

    /// Reclaim the learning from any tournament left `running` by a hard kill
    /// (k8s `activeDeadlineSeconds`, OOM, node loss): a SIGKILLed run never gets to
    /// `finish_tournament`, so its header stays `running` and its checkpointed
    /// per-team brains are stranded — the elite pool only breeds from finished runs.
    /// We hold the cross-cluster run lock here, so any `running` row for this
    /// experiment is provably a zombie (no live run but ours, which has no header
    /// yet). Flip them to `partial` so their top finishers feed the next generation.
    /// Returns the number of headers reconciled. Idempotent.
    pub fn reconcile_orphaned_running_tournaments(
        &mut self,
        experiment_id: &str,
    ) -> Result<u64, String> {
        self.ensure_connected()?;
        {
            let mut tx = self
                .client
                .transaction()
                .map_err(|err| format!("begin tournament reconcile schema tx: {err}"))?;
            ensure_soccer_tournament_tables(&mut tx)?;
            tx.commit()
                .map_err(|err| format!("commit tournament reconcile schema: {err}"))?;
        }
        let affected = self
            .client
            .execute(
                r#"
                update des_soccer_tournaments
                set status = 'partial',
                    finished_at = coalesce(finished_at, now()),
                    updated_at = now()
                where experiment_id = $1::text::uuid and status = 'running'
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("reconcile orphaned tournaments: {err}"))?;
        Ok(affected)
    }

    pub fn load_latest_active_policy(
        &mut self,
        experiment_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
    ) -> Result<Option<SoccerLearningPgPolicyVersion>, String> {
        // Full policy (every tabular entry) — what the LEARNER needs to resume exactly.
        self.load_latest_active_policy_with_min_visits(experiment_id, home_options, away_options, 0)
    }

    /// Like [`Self::load_latest_active_policy`] but returns only tabular entries with
    /// `visits >= min_visits` (the neural net is always loaded in full). The
    /// inference-only live server uses this to load just the well-learned core of the
    /// tip instead of hauling the full multi-million-row policy over the wire; pass `0`
    /// for the complete policy (training resume).
    pub fn load_latest_active_policy_with_min_visits(
        &mut self,
        experiment_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        min_visits: i32,
    ) -> Result<Option<SoccerLearningPgPolicyVersion>, String> {
        self.load_latest_policy_with_min_visits(
            experiment_id,
            home_options,
            away_options,
            min_visits,
            false,
        )
    }

    /// Like [`Self::load_latest_active_policy_with_min_visits`], but `include_unpromoted`
    /// selects the best-fitness candidate regardless of promotion status (see
    /// [`Self::load_latest_policy_metadata`]). Default paths pass `false`.
    pub fn load_latest_policy_with_min_visits(
        &mut self,
        experiment_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        min_visits: i32,
        include_unpromoted: bool,
    ) -> Result<Option<SoccerLearningPgPolicyVersion>, String> {
        self.ensure_connected()?;
        let Some(metadata) = self.load_latest_policy_metadata(experiment_id, include_unpromoted)?
        else {
            return Ok(None);
        };
        self.load_policy_version_from_metadata(metadata, home_options, away_options, min_visits)
    }

    pub fn load_policy_version_with_min_visits(
        &mut self,
        policy_version_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        min_visits: i32,
    ) -> Result<Option<SoccerLearningPgPolicyVersion>, String> {
        self.ensure_connected()?;
        let Some(metadata) = self.load_policy_metadata_by_id(policy_version_id)? else {
            return Ok(None);
        };
        self.load_policy_version_from_metadata(metadata, home_options, away_options, min_visits)
    }

    fn load_policy_version_from_metadata(
        &mut self,
        metadata: SoccerLearningPgPolicyMetadata,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        min_visits: i32,
    ) -> Result<Option<SoccerLearningPgPolicyVersion>, String> {
        let policies =
            self.load_policy_entries(&metadata.id, home_options, away_options, min_visits.max(0))?;
        let policy_fingerprint = metadata
            .policy_fingerprint
            .or_else(|| Some(soccer_team_q_policies_fingerprint(&policies)));
        let policy_entry_count = metadata
            .policy_entry_count
            .or_else(|| Some(soccer_policy_entry_count(&policies)));
        Ok(Some(SoccerLearningPgPolicyVersion {
            id: metadata.id,
            generation: metadata.generation,
            updated_at_micros: metadata.updated_at_micros,
            policy_fingerprint,
            policy_entry_count,
            policies,
            neural_network: metadata.neural_network,
            tactical_learning: metadata.tactical_learning,
            search_metadata: metadata.search_metadata,
        }))
    }

    pub fn load_latest_active_policy_metadata(
        &mut self,
        experiment_id: &str,
    ) -> Result<Option<SoccerLearningPgPolicyMetadata>, String> {
        self.load_latest_policy_metadata(experiment_id, false)
    }

    /// Latest policy metadata for a live/inference server. With `include_unpromoted =
    /// false` this is the strict "newest ACTIVE (promotion-gated)" selection every
    /// training path relies on. With `include_unpromoted = true` it returns the
    /// best-fitness candidate at the NEWEST generation that passed the learner-side
    /// promotion gate, even if the DB active-head guard later archived it. That keeps
    /// :5055 on the latest strong branch tip without resuming from windows the
    /// comparison gate explicitly rejected.
    pub fn load_latest_policy_metadata(
        &mut self,
        experiment_id: &str,
        include_unpromoted: bool,
    ) -> Result<Option<SoccerLearningPgPolicyMetadata>, String> {
        self.ensure_connected()?;
        // Same row shape for both selections; only the WHERE/ORDER differ.
        let sql = if include_unpromoted {
            r#"
                select
                  id::text,
                  generation,
                  fitness_micros,
                  metrics,
                  config,
                  coalesce((extract(epoch from updated_at) * 1000000)::bigint, 0)
                from des_soccer_learning_policy_versions
                where experiment_id = $1::text::uuid and fitness_micros is not null
                  and coalesce(
                    metrics #>> '{learningProvenance,searchParameters,promotion,status}',
                    'active'
                  ) != 'archived'
                order by generation desc, fitness_micros desc, updated_at desc, id desc
                limit 1
                "#
        } else {
            r#"
                select
                  id::text,
                  generation,
                  fitness_micros,
                  metrics,
                  config,
                  coalesce((extract(epoch from updated_at) * 1000000)::bigint, 0)
                from des_soccer_learning_policy_versions
                where experiment_id = $1::text::uuid and status = 'active'
                order by generation desc, updated_at desc, id desc
                limit 1
                "#
        };
        let Some(row) = self
            .client
            .query_opt(sql, &[&experiment_id])
            .map_err(|err| format!("select latest soccer policy version metadata: {err}"))?
        else {
            return Ok(None);
        };
        let id: String = row.get(0);
        let generation: i32 = row.get(1);
        let fitness_micros: i64 = row.get(2);
        let metrics: Value = row.get(3);
        let config: Value = row.get(4);
        let updated_at_micros: i64 = row.get(5);
        let neural_network = soccer_policy_version_neural_network_from_metrics(&metrics)?;
        let tactical_learning =
            soccer_policy_version_tactical_learning_from_values(&config, &metrics)?;
        let search_metadata = soccer_policy_version_search_metadata_from_metrics(&metrics)?;
        let policy_fingerprint = soccer_policy_version_policy_fingerprint_from_metrics(&metrics);
        let policy_entry_count = soccer_policy_version_policy_entry_count_from_metrics(&metrics);
        Ok(Some(SoccerLearningPgPolicyMetadata {
            id,
            generation,
            fitness_micros,
            updated_at_micros,
            policy_fingerprint,
            policy_entry_count,
            neural_network,
            tactical_learning,
            search_metadata,
        }))
    }

    /// Strongest recent persisted neural snapshot for warm-start/live inference.
    /// Live neural-authoritative play must not load snapshots that a promotion or
    /// anchor gate already archived; those can have high self-play fitness while
    /// losing the held-out comparison.
    pub fn load_latest_neural_policy_metadata(
        &mut self,
        experiment_id: &str,
    ) -> Result<Option<SoccerLearningPgPolicyMetadata>, String> {
        self.ensure_connected()?;
        let lookback_generations = soccer_neural_snapshot_lookback_generations();
        let min_sample_games = soccer_neural_snapshot_min_sample_games();
        let latest_qualified_trained = soccer_neural_snapshot_latest_qualified_trained();
        let Some(row) = self
            .client
            .query_opt(
                r#"
                with latest_generation as (
                  select coalesce(max(generation), 0) as generation
                  from des_soccer_learning_policy_versions
                  where experiment_id = $1::text::uuid
                    and metrics ? 'neuralNetwork'
                ),
                candidates as (
                select
                  policy_versions.id::text,
                  policy_versions.generation,
                  coalesce(policy_versions.fitness_micros, 0) as fitness_micros,
                  policy_versions.metrics,
                  policy_versions.config,
                  coalesce((extract(epoch from policy_versions.updated_at) * 1000000)::bigint, 0) as updated_at_micros,
                  nullif(policy_versions.metrics #>> '{learningProvenance,searchParameters,promotion,gate,sampleGames}', '')::int as sample_games,
                  nullif(policy_versions.metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanMatchFitness}', '')::double precision as mean_match_fitness,
                  nullif(policy_versions.metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanPlayQuality}', '')::double precision as mean_play_quality,
                  nullif(policy_versions.metrics #>> '{neuralNetwork,trainingSteps}', '')::int as neural_training_steps,
                  policy_versions.created_at,
                  policy_versions.updated_at
                from des_soccer_learning_policy_versions policy_versions
                cross join latest_generation
                where policy_versions.experiment_id = $1::text::uuid
                  and policy_versions.metrics ? 'neuralNetwork'
                  and policy_versions.generation between latest_generation.generation - $2 and latest_generation.generation
                  and (
                    policy_versions.status = 'active'
                    or (
                      (policy_versions.metrics #>> '{learningProvenance,searchParameters,neuralCheckpoint,reason}') is null
                      and coalesce(
                        policy_versions.metrics #>> '{learningProvenance,searchParameters,promotion,status}',
                        'active'
                      ) != 'archived'
                      and coalesce(
                        policy_versions.metrics #>> '{learningProvenance,searchParameters,anchorPromotion,status}',
                        'active'
                      ) != 'archived'
                    )
                  )
                )
                select
                  id,
                  generation,
                  fitness_micros,
                  metrics,
                  config,
                  updated_at_micros
                from candidates
                order by
                  case
                    when $4::bool
                      and sample_games >= $3
                      and coalesce(neural_training_steps, 0) > 0
                    then 0
                    when not $4::bool
                      and sample_games >= $3
                      and mean_match_fitness is not null
                      and mean_play_quality is not null
                    then 0
                    else 1
                  end,
                  case when $4::bool then mean_match_fitness end desc nulls last,
                  case when $4::bool then mean_play_quality end desc nulls last,
                  case when $4::bool then sample_games end desc nulls last,
                  case when $4::bool then neural_training_steps end desc nulls last,
                  case when $4::bool then generation end desc nulls last,
                  case when not $4::bool then mean_match_fitness end desc nulls last,
                  case when not $4::bool then mean_play_quality end desc nulls last,
                  generation desc,
                  created_at desc,
                  updated_at desc,
                  id desc
                limit 1
                "#,
                &[
                    &experiment_id,
                    &lookback_generations,
                    &min_sample_games,
                    &latest_qualified_trained,
                ],
            )
            .map_err(|err| format!("select latest soccer neural policy metadata: {err}"))?
        else {
            return Ok(None);
        };
        let id: String = row.get(0);
        let generation: i32 = row.get(1);
        let fitness_micros: i64 = row.get(2);
        let metrics: Value = row.get(3);
        let config: Value = row.get(4);
        let updated_at_micros: i64 = row.get(5);
        let neural_network = soccer_policy_version_neural_network_from_metrics(&metrics)?;
        let tactical_learning =
            soccer_policy_version_tactical_learning_from_values(&config, &metrics)?;
        let search_metadata = soccer_policy_version_search_metadata_from_metrics(&metrics)?;
        let policy_fingerprint = soccer_policy_version_policy_fingerprint_from_metrics(&metrics);
        let policy_entry_count = soccer_policy_version_policy_entry_count_from_metrics(&metrics);
        Ok(Some(SoccerLearningPgPolicyMetadata {
            id,
            generation,
            fitness_micros,
            updated_at_micros,
            policy_fingerprint,
            policy_entry_count,
            neural_network,
            tactical_learning,
            search_metadata,
        }))
    }

    pub fn load_policy_metadata_by_id(
        &mut self,
        policy_version_id: &str,
    ) -> Result<Option<SoccerLearningPgPolicyMetadata>, String> {
        self.ensure_connected()?;
        let Some(row) = self
            .client
            .query_opt(
                r#"
                select
                  id::text,
                  generation,
                  fitness_micros,
                  metrics,
                  config,
                  coalesce((extract(epoch from updated_at) * 1000000)::bigint, 0)
                from des_soccer_learning_policy_versions
                where id = $1::text::uuid
                limit 1
                "#,
                &[&policy_version_id],
            )
            .map_err(|err| format!("select soccer policy version metadata by id: {err}"))?
        else {
            return Ok(None);
        };
        let id: String = row.get(0);
        let generation: i32 = row.get(1);
        let fitness_micros: i64 = row.get(2);
        let metrics: Value = row.get(3);
        let config: Value = row.get(4);
        let updated_at_micros: i64 = row.get(5);
        let neural_network = soccer_policy_version_neural_network_from_metrics(&metrics)?;
        let tactical_learning =
            soccer_policy_version_tactical_learning_from_values(&config, &metrics)?;
        let search_metadata = soccer_policy_version_search_metadata_from_metrics(&metrics)?;
        let policy_fingerprint = soccer_policy_version_policy_fingerprint_from_metrics(&metrics);
        let policy_entry_count = soccer_policy_version_policy_entry_count_from_metrics(&metrics);
        Ok(Some(SoccerLearningPgPolicyMetadata {
            id,
            generation,
            fitness_micros,
            updated_at_micros,
            policy_fingerprint,
            policy_entry_count,
            neural_network,
            tactical_learning,
            search_metadata,
        }))
    }

    pub fn load_strongest_recent_policy_promotion_baseline(
        &mut self,
        experiment_id: &str,
        latest_generation: i32,
        lookback_generations: i32,
        min_sample_games: usize,
    ) -> Result<Option<SoccerLearningPgPolicyPromotionBaseline>, String> {
        let min_generation = latest_generation.saturating_sub(lookback_generations.max(0));
        let min_sample_games = checked_i32(min_sample_games);
        let row = self.with_transient_retry(
            "select strongest soccer promotion baseline",
            |store| {
                store.client.query_opt(
                    r#"
                select
                  id::text,
                  generation,
                  (metrics #>> '{learningProvenance,searchParameters,promotion,gate,sampleGames}')::int,
                  (metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanMatchFitness}')::double precision,
                  (metrics #>> '{learningProvenance,searchParameters,promotion,gate,bestMatchFitness}')::double precision,
                  (metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanPlayQuality}')::double precision
                from des_soccer_learning_policy_versions
                where experiment_id = $1::text::uuid
                  and generation between $2 and $3
                  and coalesce(
                    metrics #>> '{learningProvenance,searchParameters,promotion,status}',
                    'archived'
                  ) = 'active'
                  and (metrics #>> '{learningProvenance,searchParameters,promotion,gate,sampleGames}')::int >= $4
                  and (metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanMatchFitness}') is not null
                  and (metrics #>> '{learningProvenance,searchParameters,promotion,gate,bestMatchFitness}') is not null
                  and (metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanPlayQuality}') is not null
                order by
                  (metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanMatchFitness}')::double precision desc,
                  (metrics #>> '{learningProvenance,searchParameters,promotion,gate,meanPlayQuality}')::double precision desc,
                  generation desc,
                  updated_at desc,
                  id desc
                limit 1
                "#,
                &[
                    &experiment_id,
                    &min_generation,
                    &latest_generation,
                    &min_sample_games,
                ],
                )
                .map_err(|err| format!("select strongest soccer promotion baseline: {err}"))
            },
        )?;
        let Some(row) = row else {
            return Ok(None);
        };
        let sample_games: i32 = row.get(2);
        Ok(Some(SoccerLearningPgPolicyPromotionBaseline {
            policy_version_id: row.get(0),
            generation: row.get(1),
            sample_games: usize::try_from(sample_games.max(0)).unwrap_or(usize::MAX),
            mean_match_fitness: row.get(3),
            best_match_fitness: row.get(4),
            mean_play_quality: row.get(5),
        }))
    }

    pub fn insert_policy_version(
        &mut self,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
    ) -> Result<String, String> {
        let policy_version_id = Uuid::new_v4().to_string();
        self.insert_policy_version_with_id(
            &policy_version_id,
            experiment_id,
            parent_policy_version_id,
            generation,
            version_label,
            source_kind,
            status,
            config,
            home_options,
            away_options,
            policies,
            fitness,
        )?;
        Ok(policy_version_id)
    }

    pub fn insert_policy_version_with_id(
        &mut self,
        policy_version_id: &str,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
    ) -> Result<(), String> {
        self.insert_policy_version_with_id_inner(
            policy_version_id,
            experiment_id,
            parent_policy_version_id,
            generation,
            version_label,
            source_kind,
            status,
            config,
            home_options,
            away_options,
            policies,
            fitness,
            None,
            None,
        )
    }

    pub fn insert_policy_version_with_id_and_neural_network(
        &mut self,
        policy_version_id: &str,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
        neural_network: Option<&SoccerNeuralNetworkSnapshot>,
    ) -> Result<(), String> {
        self.insert_policy_version_with_id_inner(
            policy_version_id,
            experiment_id,
            parent_policy_version_id,
            generation,
            version_label,
            source_kind,
            status,
            config,
            home_options,
            away_options,
            policies,
            fitness,
            neural_network,
            None,
        )
    }

    pub fn insert_policy_version_with_id_and_neural_network_and_search_metadata(
        &mut self,
        policy_version_id: &str,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
        neural_network: Option<&SoccerNeuralNetworkSnapshot>,
        search_metadata: Option<&Value>,
    ) -> Result<(), String> {
        self.insert_policy_version_with_id_inner(
            policy_version_id,
            experiment_id,
            parent_policy_version_id,
            generation,
            version_label,
            source_kind,
            status,
            config,
            home_options,
            away_options,
            policies,
            fitness,
            neural_network,
            search_metadata,
        )
    }

    fn insert_policy_version_with_id_inner(
        &mut self,
        policy_version_id: &str,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
        neural_network: Option<&SoccerNeuralNetworkSnapshot>,
        search_metadata: Option<&Value>,
    ) -> Result<(), String> {
        self.ensure_connected()?;
        let config_json =
            serde_json::to_value(config).map_err(|err| format!("serialize match config: {err}"))?;
        let options_json = json!({
            "home": home_options,
            "away": away_options,
        });
        let lineage = parent_policy_version_id
            .map(|id| json!([id]))
            .unwrap_or_else(|| json!([]));
        let mut base_metrics = soccer_policy_version_metrics(
            source_kind,
            fitness,
            neural_network,
            Some(&config.tactical_learning),
            search_metadata,
        )?;
        stamp_soccer_policy_weight_metrics(&mut base_metrics, policies);
        let retention_kind = SOCCER_POLICY_RETENTION_BRANCH_TIP;
        let entry_count =
            checked_i32(policies.home.entries().len() + policies.away.entries().len());
        let target_entry_count = checked_i32(
            policies.home.target_entries().len() + policies.away.target_entries().len(),
        );
        let visit_count = checked_i64(policies.home.visit_count() + policies.away.visit_count());
        let fitness_micros = soccer_learning_to_micros(fitness);
        self.ensure_policy_retention_schema_ready()?;
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer policy version transaction: {err}"))?;
        let branch_key = soccer_policy_branch_key_for_insert(
            &mut tx,
            parent_policy_version_id,
            policy_version_id,
        )?;
        let metrics_for_status = soccer_policy_version_metrics_with_retention(
            base_metrics.clone(),
            policy_version_id,
            &branch_key,
            retention_kind,
            soccer_policy_version_retains_full_entries(status),
        );
        let effective_status =
            soccer_policy_version_status_after_promotion_sample_floor(status, &metrics_for_status);
        let full_entries_retained = soccer_policy_version_retains_full_entries(effective_status)
            && soccer_policy_version_full_entries_enabled();
        base_metrics["learningProvenance"]["policyWeightsPersisted"] = json!(full_entries_retained);
        let metrics = soccer_policy_version_metrics_with_retention(
            base_metrics,
            policy_version_id,
            &branch_key,
            retention_kind,
            full_entries_retained,
        );

        if effective_status == SOCCER_POLICY_STATUS_ACTIVE {
            // Serialize active-policy promotions PER EXPERIMENT so the
            // archive-old-active + insert-new-active pair is atomic against any
            // other promoter (a concurrent tournament run, the continuous learner,
            // a second replica). Without it, under READ COMMITTED two promoters can
            // each archive the other's not-yet-committed active and both insert,
            // leaving TWO rows at status='active'. This transaction-scoped lock is
            // auto-released on commit/rollback and only contends writers promoting
            // the SAME experiment (different experiments hash to different keys, so
            // the common single-writer path sees zero contention). Every promoter
            // funnels through this function, so locking here covers them all.
            tx.execute(
                "select pg_advisory_xact_lock(hashtext('des-soccer-policy-active-promote'), hashtext($1))",
                &[&experiment_id],
            )
            .map_err(|err| format!("acquire policy promotion advisory lock: {err}"))?;
            tx.execute(
                "select set_config('des_soccer.policy_active_promotion_in_progress', '1', true)",
                &[],
            )
            .map_err(|err| format!("mark policy promotion transaction: {err}"))?;
            tx.execute(
                r#"
                update des_soccer_learning_policy_versions
                set status = 'archived', updated_at = now()
                where experiment_id = $1::text::uuid and status = 'active'
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("archive old soccer policy versions: {err}"))?;
        }

        let inserted = tx
            .execute(
                r#"
            insert into des_soccer_learning_policy_versions
              (
                id,
                experiment_id,
                parent_policy_version_id,
                generation,
                version_label,
                source_kind,
                status,
                options,
                config,
                lineage,
                metrics,
                entry_count,
                target_entry_count,
                visit_count,
                fitness_micros,
                branch_key,
                retention_kind,
                full_entries_retained
              )
            values
              (
                $1::text::uuid,
                $2::text::uuid,
                $3::text::uuid,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16::text::uuid,
                $17,
                $18
              )
            "#,
                &[
                    &policy_version_id,
                    &experiment_id,
                    &parent_policy_version_id,
                    &generation,
                    &version_label,
                    &source_kind,
                    &effective_status,
                    &options_json,
                    &config_json,
                    &lineage,
                    &metrics,
                    &entry_count,
                    &target_entry_count,
                    &visit_count,
                    &fitness_micros,
                    &branch_key,
                    &retention_kind,
                    &full_entries_retained,
                ],
            )
            .map_err(|err| {
                format!(
                    "insert soccer policy version: {}",
                    soccer_learning_pg_error_context(&err)
                )
            })?;
        if inserted != 1 {
            return Err(format!(
                "insert soccer policy version inserted {inserted} rows for policy version {policy_version_id}"
            ));
        }

        if full_entries_retained {
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Home,
                &policies.home,
                None,
            )?;
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Away,
                &policies.away,
                None,
            )?;
            // NOTE: the superseded generation's entries are deliberately NOT pruned inside this
            // transaction. Doing so meant a multi-million-row DELETE of the previous tip's
            // entries had to complete within the session `statement_timeout`, or the ENTIRE
            // promotion rolled back — which stalled generation promotion for a week. Cleanup now
            // runs best-effort AFTER the commit below.
        }

        tx.commit()
            .map_err(|err| format!("commit soccer policy version: {err}"))?;

        // The promotion is now durable. Cleanup is deliberately out-of-band by default: on the
        // production policy table, even LIMIT-bounded deletes can block learners for minutes while
        // autovacuum and heap reads churn. Operators can opt in for a dedicated maintenance run.
        if full_entries_retained {
            if soccer_policy_inline_prune_enabled() {
                if let Err(err) = self.prune_superseded_branch_entries_batched(
                    experiment_id,
                    &branch_key,
                    policy_version_id,
                ) {
                    eprintln!(
                        "soccer policy retention: superseded-entry cleanup deferred (best-effort) for {policy_version_id}: {err}"
                    );
                }
            } else {
                eprintln!(
                    "soccer policy retention: superseded-entry cleanup deferred for {policy_version_id}; set {SOCCER_POLICY_INLINE_PRUNE_ENV}=true for dedicated inline maintenance"
                );
            }
        }
        Ok(())
    }

    /// Best-effort, timeout-bounded cleanup of every NON-tip version's entries on a branch, run
    /// AFTER a promotion has committed (see [`Self::insert_policy_version_with_id_inner`]).
    /// Deletes in [`SOCCER_POLICY_PRUNE_DELETE_BATCH_SIZE`] chunks — each its own statement,
    /// comfortably under the session `statement_timeout`, so a multi-million-row cleanup never
    /// trips it — then flips the drained versions to `full_entries_retained = false`. Safe to
    /// interrupt or rerun: a version stays retained until its entries are gone, so an aborted run
    /// is picked up by the next promotion. Returns the number of entries deleted.
    fn prune_superseded_branch_entries_batched(
        &mut self,
        experiment_id: &str,
        branch_key: &str,
        tip_policy_version_id: &str,
    ) -> Result<u64, String> {
        self.ensure_connected()?;
        // Resolve the superseded version ids FIRST (a tiny scan of the small policy-versions
        // table). Deleting by `policy_version_id` then drives that column's index directly,
        // instead of a join-scan over millions of entry rows per batch — which blew the
        // statement_timeout and was the cleanup's "db error". Ids are carried as text (the
        // `postgres` dep has no `with-uuid` feature) and cast back to uuid in the predicates.
        let superseded: Vec<String> = self
            .client
            .query(
                r#"
                select id::text
                from des_soccer_learning_policy_versions
                where experiment_id = $1::text::uuid
                  and branch_key = $2::text::uuid
                  and id <> $3::text::uuid
                  and full_entries_retained = true
                "#,
                &[&experiment_id, &branch_key, &tip_policy_version_id],
            )
            .map_err(|err| format!("list superseded branch versions: {err}"))?
            .iter()
            .map(|row| row.get::<_, String>(0))
            .collect();
        if superseded.is_empty() {
            return Ok(0);
        }
        // Disable the 30s statement_timeout for the cleanup: each batch is LIMIT-bounded and this
        // is best-effort + post-commit, but later batches slow as dead tuples accumulate and were
        // tripping the 30s cap (deleting one batch, then erroring -> the superseded gen never
        // drained). Restored below on EVERY path so other queries on this shared connection keep
        // their guard. Errors are captured (not `?`-propagated) so the restore always runs.
        let _ = self.client.batch_execute("SET statement_timeout = 0");
        let mut total: u64 = 0;
        let mut outcome: Result<(), String> = Ok(());
        loop {
            match self.client.execute(
                r#"
                    delete from des_soccer_learning_policy_entries
                    where ctid in (
                      select ctid
                      from des_soccer_learning_policy_entries
                      where policy_version_id = any($1::text[]::uuid[])
                      limit $2
                    )
                    "#,
                &[&superseded, &SOCCER_POLICY_PRUNE_DELETE_BATCH_SIZE],
            ) {
                Ok(deleted) => {
                    total += deleted;
                    if deleted == 0 {
                        break;
                    }
                }
                Err(err) => {
                    outcome = Err(format!("prune superseded branch entries (batch): {err}"));
                    break;
                }
            }
        }
        if outcome.is_ok() {
            // Entries are gone — flip the drained versions to not-retained (cheap; the
            // policy-versions table is tiny). Mirrors the old in-transaction prune's bookkeeping.
            if let Err(err) = self.client.execute(
                r#"
                update des_soccer_learning_policy_versions
                set
                  full_entries_retained = false,
                  full_entries_pruned_at = now(),
                  updated_at = now(),
                  metrics = jsonb_set(
                    metrics,
                    '{postgresRetention}',
                    coalesce(metrics->'postgresRetention', '{}'::jsonb)
                      || jsonb_build_object(
                        'fullEntriesRetained', false,
                        'prunedByPolicyVersionId', $2,
                        'retentionKind', 'branch_tip'
                      ),
                    true
                  )
                where id = any($1::text[]::uuid[])
                "#,
                &[&superseded, &tip_policy_version_id],
            ) {
                outcome = Err(format!("mark superseded branch versions pruned: {err}"));
            }
        }
        // Always restore the connection's statement_timeout guard.
        let _ = self.client.batch_execute(&format!(
            "SET statement_timeout = '{}'",
            soccer_learning_pg_statement_timeout()
        ));
        outcome.map(|()| total)
    }

    pub fn insert_completed_run(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        base_policy_version_id: Option<&str>,
        output_policy_version_id: Option<&str>,
        game: &SoccerLearningCompletedGame,
    ) -> Result<String, String> {
        self.ensure_connected()?;
        if !game.config_moments.is_empty() {
            validate_config_moments(&game.config_moments)?;
        }
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer run transaction: {err}"))?;
        let run_id = insert_completed_run_in_transaction(
            &mut tx,
            experiment_id,
            runner_id,
            base_policy_version_id,
            output_policy_version_id,
            game,
        )?;
        if !game.config_moments.is_empty() {
            ensure_soccer_config_moment_tables(&mut tx)?;
            insert_config_moments_in_transaction(
                &mut tx,
                Some(run_id.as_str()),
                Some(experiment_id),
                &game.config_moments,
            )?;
            soft_delete_old_moment_vectors_in_transaction(
                &mut tx,
                Some(experiment_id),
                SOCCER_MOMENT_VECTOR_SOFT_DELETE_AFTER_DAYS,
            )?;
        }
        if !game.pass_outcome_samples.is_empty() {
            ensure_soccer_pass_outcome_tables(&mut tx)?;
            insert_pass_outcome_samples_in_transaction(
                &mut tx,
                Some(run_id.as_str()),
                Some(experiment_id),
                &game.pass_outcome_samples,
            )?;
        }
        tx.commit()
            .map_err(|err| format!("commit soccer learning run: {err}"))?;
        Ok(run_id)
    }

    /// Load the most recent pass-outcome training samples for an experiment (newest first), the
    /// corpus the cluster learner trains [`SoccerPassCompletionHead`] on. `limit` caps the batch.
    pub fn load_pass_outcome_samples(
        &mut self,
        experiment_id: &str,
        limit: i64,
    ) -> Result<Vec<SoccerPassOutcomeSample>, String> {
        self.ensure_connected()?;
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin pass-outcome load transaction: {err}"))?;
        ensure_soccer_pass_outcome_tables(&mut tx)?;
        let rows = tx
            .query(
                "select features, completed, own_half \
                 from des_soccer_pass_outcome_samples \
                 where experiment_id = $1::text::uuid and deleted_at is null \
                 order by created_at desc limit $2",
                &[&experiment_id, &limit],
            )
            .map_err(|err| format!("load soccer pass-outcome samples: {err}"))?;
        let samples = rows
            .into_iter()
            .map(|row| {
                let features: Vec<f32> = row.get(0);
                SoccerPassOutcomeSample {
                    features,
                    completed: row.get(1),
                    own_half: row.get(2),
                }
            })
            .collect();
        tx.commit()
            .map_err(|err| format!("commit pass-outcome load: {err}"))?;
        Ok(samples)
    }

    /// Persist moment embeddings (one fixed-width vector per transition) for
    /// cross-game similarity retrieval (RAG). `run_id` / `experiment_id` are
    /// optional provenance (validated as uuids by the cast). Embeddings whose
    /// width != [`SOCCER_MOMENT_EMBEDDING_DIM`] are rejected, since the
    /// `vector(N)` column is fixed. Returns the number of rows written.
    pub fn insert_moment_embeddings(
        &mut self,
        run_id: Option<&str>,
        experiment_id: Option<&str>,
        moments: &[SoccerMomentEmbeddingInsert],
    ) -> Result<usize, String> {
        if moments.is_empty() {
            return Ok(0);
        }
        for moment in moments {
            if moment.embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM {
                return Err(format!(
                    "moment embedding has {} dims, expected {SOCCER_MOMENT_EMBEDDING_DIM}",
                    moment.embedding.len()
                ));
            }
        }
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin moment embedding transaction: {err}"))?;
        ensure_soccer_moment_embedding_tables(&mut tx)?;
        // Chunked multi-row insert: one statement per chunk rather than a round
        // trip per moment (a full game produces thousands). `run_id`/`experiment_id`
        // are shared $1/$2 reused by every row, so each row adds only 6 params —
        // `MOMENT_EMBEDDING_INSERT_CHUNK_ROWS` keeps the total far under Postgres's
        // 65535-parameter ceiling.
        for chunk in moments.chunks(MOMENT_EMBEDDING_INSERT_CHUNK_ROWS) {
            // Own the per-row column data so the `&dyn ToSql` refs outlive execute.
            let teams: Vec<&'static str> =
                chunk.iter().map(|m| soccer_team_label(m.team)).collect();
            let ticks: Vec<i64> = chunk.iter().map(|m| checked_i64(m.tick)).collect();
            let rewards: Vec<i64> = chunk
                .iter()
                .map(|m| soccer_learning_to_micros(m.reward))
                .collect();
            let values: Vec<Option<i64>> = chunk
                .iter()
                .map(|m| m.value.map(soccer_learning_to_micros))
                .collect();
            let embeddings: Vec<String> =
                chunk.iter().map(|m| pg_vector_text(&m.embedding)).collect();
            let sql = format!(
                "insert into des_soccer_moment_embeddings \
                 (run_id, experiment_id, team, tick, action, reward_micros, value_micros, embedding) \
                 values {}",
                moment_embedding_values_clause(chunk.len())
            );
            let mut params: Vec<&(dyn ToSql + Sync)> = vec![&run_id, &experiment_id];
            for (i, moment) in chunk.iter().enumerate() {
                params.push(&teams[i]);
                params.push(&ticks[i]);
                params.push(&moment.action);
                params.push(&rewards[i]);
                params.push(&values[i]);
                params.push(&embeddings[i]);
            }
            tx.execute(&sql, &params)
                .map_err(|err| format!("insert soccer moment embeddings: {err}"))?;
        }
        soft_delete_old_moment_vectors_in_transaction(
            &mut tx,
            experiment_id,
            SOCCER_MOMENT_VECTOR_SOFT_DELETE_AFTER_DAYS,
        )?;
        tx.commit()
            .map_err(|err| format!("commit soccer moment embeddings: {err}"))?;
        Ok(moments.len())
    }

    /// Approximate-nearest-neighbour retrieval over stored moments by cosine
    /// distance to `query` (RAG read path). Optionally restrict to one `team`.
    /// Returns up to `limit` neighbours ordered nearest-first, each with its
    /// distance and the outcome metadata a consumer ranks on.
    pub fn search_nearest_moments(
        &mut self,
        query: &[f64],
        limit: usize,
        team: Option<Team>,
    ) -> Result<Vec<SoccerMomentNeighbor>, String> {
        if query.len() != SOCCER_MOMENT_EMBEDDING_DIM {
            return Err(format!(
                "moment query has {} dims, expected {SOCCER_MOMENT_EMBEDDING_DIM}",
                query.len()
            ));
        }
        // Reconnect if the session dropped since the last op — every other DB
        // method does this first; without it a dropped connection fails hard here
        // instead of transparently reconnecting.
        self.ensure_connected()?;
        // Ensure the schema in a COMMITTED transaction (so the table actually
        // exists if nothing has been written yet) — the previous form dropped the
        // tx, silently rolling the `create table` back.
        {
            let mut tx = self
                .client
                .transaction()
                .map_err(|err| format!("begin moment search schema tx: {err}"))?;
            ensure_soccer_moment_embedding_tables(&mut tx)?;
            tx.commit()
                .map_err(|err| format!("commit moment search schema: {err}"))?;
        }
        let limit = limit.clamp(1, 1000) as i64;
        let query_text = pg_vector_text(query);
        let team_filter = team.map(soccer_team_label);
        let rows = self
            .client
            .query(
                SOCCER_MOMENT_EMBEDDING_SEARCH_SQL,
                &[&query_text, &team_filter, &limit],
            )
            .map_err(|err| format!("search soccer moment embeddings: {err}"))?;
        Ok(rows
            .iter()
            .map(|row| SoccerMomentNeighbor {
                team: row.get::<_, String>(0),
                action: row.get::<_, String>(1),
                reward: soccer_learning_from_micros(row.get::<_, i64>(2)),
                value: row
                    .get::<_, Option<i64>>(3)
                    .map(soccer_learning_from_micros),
                tick: row.get::<_, i64>(4),
                distance: row.get::<_, f64>(5),
            })
            .collect())
    }

    /// Persist whole-field **configuration moments** (the retrieval corpus): one
    /// row per captured ball-carrier decision, carrying the config embedding (for
    /// ANN search), the raw canonical features (for an exact permutation re-score),
    /// the decision taken, and the outcome (immediate reward + n-step return).
    /// Embeddings whose width != [`SOCCER_MOMENT_EMBEDDING_DIM`] are rejected.
    /// Returns the number of rows written.
    pub fn insert_config_moments(
        &mut self,
        run_id: Option<&str>,
        experiment_id: Option<&str>,
        moments: &[SoccerConfigMomentInsert],
    ) -> Result<usize, String> {
        if moments.is_empty() {
            return Ok(0);
        }
        validate_config_moments(moments)?;
        self.ensure_connected()?;
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin config moment transaction: {err}"))?;
        ensure_soccer_config_moment_tables(&mut tx)?;
        insert_config_moments_in_transaction(&mut tx, run_id, experiment_id, moments)?;
        soft_delete_old_moment_vectors_in_transaction(
            &mut tx,
            experiment_id,
            SOCCER_MOMENT_VECTOR_SOFT_DELETE_AFTER_DAYS,
        )?;
        tx.commit()
            .map_err(|err| format!("commit soccer config moments: {err}"))?;
        Ok(moments.len())
    }

    /// Approximate-nearest-neighbour retrieval over stored configuration moments
    /// by cosine distance to `query` (the live config embedding). Optionally
    /// restrict to one `team`. Returns up to `limit` neighbours nearest-first,
    /// each with its decision, outcome, raw features (for re-score), and distance.
    /// This is the "find the N closest configurations" step.
    pub fn search_nearest_config_moments(
        &mut self,
        query: &[f64],
        limit: usize,
        team: Option<Team>,
        comparison: SoccerConfigComparison,
    ) -> Result<Vec<SoccerConfigMomentNeighbor>, String> {
        if query.len() != SOCCER_MOMENT_EMBEDDING_DIM {
            return Err(format!(
                "config moment query has {} dims, expected {SOCCER_MOMENT_EMBEDDING_DIM}",
                query.len()
            ));
        }
        // Reconnect if the session dropped since the last op (consistent with every
        // other DB method) before opening the schema-ensure transaction.
        self.ensure_connected()?;
        {
            let mut tx = self
                .client
                .transaction()
                .map_err(|err| format!("begin config moment search schema tx: {err}"))?;
            ensure_soccer_config_moment_tables(&mut tx)?;
            tx.commit()
                .map_err(|err| format!("commit config moment search schema: {err}"))?;
        }
        let limit = limit.clamp(1, 1000) as i64;
        let query_text = pg_vector_text(query);
        let team_filter = team.map(soccer_team_label);
        let search_sql = match comparison {
            SoccerConfigComparison::PositionAgnostic => SOCCER_CONFIG_MOMENT_SEARCH_SQL,
            SoccerConfigComparison::AssignedPosition => SOCCER_CONFIG_MOMENT_SEARCH_ASSIGNED_SQL,
        };
        let rows = self
            .client
            .query(search_sql, &[&query_text, &team_filter, &limit])
            .map_err(|err| format!("search soccer config moments: {err}"))?;
        Ok(rows
            .iter()
            .map(|row| SoccerConfigMomentNeighbor {
                team: row.get::<_, String>(0),
                role: row.get::<_, String>(1),
                action: row.get::<_, String>(2),
                reward: soccer_learning_from_micros(row.get::<_, i64>(3)),
                nstep_return: soccer_learning_from_micros(row.get::<_, i64>(4)),
                value: row
                    .get::<_, Option<i64>>(5)
                    .map(soccer_learning_from_micros),
                tick: row.get::<_, i64>(6),
                features: row.get::<_, Vec<f32>>(7),
                distance: row.get::<_, f64>(8),
            })
            .collect())
    }

    pub fn insert_completed_runs(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        runs: &[SoccerLearningPgCompletedRunInsert<'_>],
    ) -> Result<Vec<String>, String> {
        if runs.is_empty() {
            return Ok(Vec::new());
        }
        self.ensure_connected()?;
        let has_config_moments = runs.iter().any(|run| !run.game.config_moments.is_empty());
        let has_pass_outcome_samples = runs
            .iter()
            .any(|run| !run.game.pass_outcome_samples.is_empty());
        if has_config_moments {
            for run in runs {
                validate_config_moments(&run.game.config_moments)?;
            }
        }

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer run batch transaction: {err}"))?;
        if has_config_moments {
            ensure_soccer_config_moment_tables(&mut tx)?;
        }
        if has_pass_outcome_samples {
            ensure_soccer_pass_outcome_tables(&mut tx)?;
        }
        let mut run_ids = Vec::with_capacity(runs.len());
        for chunk in runs.chunks(SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE) {
            let chunk_run_ids = insert_completed_run_headers_in_transaction(
                &mut tx,
                experiment_id,
                runner_id,
                chunk,
            )?;
            if soccer_learning_pg_persist_run_deltas_enabled() {
                insert_completed_run_delta_rows_in_transaction(&mut tx, &chunk_run_ids, chunk)?;
            }
            if has_config_moments {
                for (run_id, run) in chunk_run_ids.iter().zip(chunk.iter()) {
                    if !run.game.config_moments.is_empty() {
                        insert_config_moments_in_transaction(
                            &mut tx,
                            Some(run_id.as_str()),
                            Some(experiment_id),
                            &run.game.config_moments,
                        )?;
                    }
                }
            }
            if has_pass_outcome_samples {
                for (run_id, run) in chunk_run_ids.iter().zip(chunk.iter()) {
                    if !run.game.pass_outcome_samples.is_empty() {
                        insert_pass_outcome_samples_in_transaction(
                            &mut tx,
                            Some(run_id.as_str()),
                            Some(experiment_id),
                            &run.game.pass_outcome_samples,
                        )?;
                    }
                }
            }
            run_ids.extend(chunk_run_ids);
        }
        if has_config_moments {
            soft_delete_old_moment_vectors_in_transaction(
                &mut tx,
                Some(experiment_id),
                SOCCER_MOMENT_VECTOR_SOFT_DELETE_AFTER_DAYS,
            )?;
        }
        tx.commit()
            .map_err(|err| format!("commit soccer learning run batch: {err}"))?;
        Ok(run_ids)
    }

    pub fn load_recent_completed_run_policy_delta(
        &mut self,
        experiment_id: &str,
        max_delta_rows: usize,
        created_after_micros: Option<i64>,
    ) -> Result<SoccerLearningPolicyDelta, String> {
        if max_delta_rows == 0 {
            return Ok(SoccerLearningPolicyDelta::default());
        }
        self.ensure_connected()?;
        let limit = max_delta_rows.min(i64::MAX as usize) as i64;
        let rows = self
            .client
            .query(
                r#"
                select
                  d.team,
                  d.entry_kind,
                  d.state_hash,
                  d.state_key,
                  d.action,
                  d.target_fine_cell_id,
                  d.target_tactical_cell_id,
                  d.target_macro_cell_id,
                  d.target_root_cell_id,
                  d.before_value_micros,
                  d.after_value_micros,
                  d.value_delta_micros,
                  d.visit_delta,
                  d.merge_weight_micros,
                  d.effective_visit_micros,
                  d.receiver_descriptor
                from des_soccer_learning_run_deltas d
                join des_soccer_learning_runs r on r.id = d.run_id
                where r.experiment_id = $1::text::uuid
                  and r.status = 'completed'
                  and ($3::bigint is null
                       or r.created_at > to_timestamp(($3::double precision) / 1000000.0))
                order by r.created_at desc, r.id desc,
                         d.team, d.entry_kind, d.state_hash, d.action,
                         d.target_fine_cell_id, d.target_tactical_cell_id,
                         d.target_macro_cell_id, d.target_root_cell_id,
                         d.receiver_descriptor
                limit $2
                "#,
                &[&experiment_id, &limit, &created_after_micros],
            )
            .map_err(|err| format!("load recent soccer completed-run policy deltas: {err}"))?;
        let mut entries = Vec::with_capacity(rows.len());
        for row in rows {
            let team_label: String = row.get(0);
            let entry_kind_label: String = row.get(1);
            let stored_state_hash: String = row.get(2);
            let state_json: Value = row.get(3);
            let state_key: SoccerQStateKey = serde_json::from_value(state_json.clone())
                .map_err(|err| format!("decode soccer run-delta state key: {err}"))?;
            let before_value_micros: i64 = row.get(9);
            let after_value_micros: i64 = row.get(10);
            let value_delta_micros: i64 = row.get(11);
            let merge_weight_micros: i64 = row.get(13);
            let effective_visit_micros: i64 = row.get(14);
            let state_hash = if stored_state_hash.trim().is_empty() {
                state_hash(&state_json)
            } else {
                stored_state_hash
            };
            entries.push(SoccerLearningPolicyDeltaEntry {
                team: soccer_team_from_label(&team_label)?,
                entry_kind: soccer_policy_entry_kind_from_label(&entry_kind_label)?,
                state_hash,
                state_key,
                state_json,
                action: row.get(4),
                target_fine_cell_id: row.get(5),
                target_tactical_cell_id: row.get(6),
                target_macro_cell_id: row.get(7),
                target_root_cell_id: row.get(8),
                before_value: soccer_learning_from_micros(before_value_micros),
                after_value: soccer_learning_from_micros(after_value_micros),
                value_delta: soccer_learning_from_micros(value_delta_micros),
                before_value_micros,
                after_value_micros,
                value_delta_micros,
                visit_delta: row.get::<_, i32>(12).max(0) as u32,
                merge_weight: soccer_learning_from_micros(merge_weight_micros),
                merge_weight_micros,
                effective_visit_weight: soccer_learning_from_micros(effective_visit_micros),
                effective_visit_micros,
            });
        }
        Ok(SoccerLearningPolicyDelta { entries })
    }

    pub fn prune_completed_runs_for_experiment(
        &mut self,
        experiment_id: &str,
        keep_latest_runs: usize,
    ) -> Result<SoccerLearningPgCompletedRunRetentionPrune, String> {
        self.ensure_connected()?;
        let Some(keep_latest_runs) = soccer_completed_run_retention_limit(keep_latest_runs) else {
            return Ok(SoccerLearningPgCompletedRunRetentionPrune::default());
        };

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer completed-run retention transaction: {err}"))?;
        let deleted_delta_rows = tx
            .execute(
                r#"
                with stale_runs as (
                  select id
                  from (
                    select
                      id,
                      row_number() over (order by created_at desc, id desc) as keep_rank
                    from des_soccer_learning_runs
                    where experiment_id = $1::text::uuid
                  ) ranked
                  where keep_rank > $2
                )
                delete from des_soccer_learning_run_deltas deltas
                using stale_runs
                where deltas.run_id = stale_runs.id
                "#,
                &[&experiment_id, &keep_latest_runs],
            )
            .map_err(|err| format!("prune soccer completed-run deltas: {err}"))?;
        let deleted_runs = tx
            .execute(
                r#"
                with stale_runs as (
                  select id
                  from (
                    select
                      id,
                      row_number() over (order by created_at desc, id desc) as keep_rank
                    from des_soccer_learning_runs
                    where experiment_id = $1::text::uuid
                  ) ranked
                  where keep_rank > $2
                )
                delete from des_soccer_learning_runs runs
                using stale_runs
                where runs.id = stale_runs.id
                "#,
                &[&experiment_id, &keep_latest_runs],
            )
            .map_err(|err| format!("prune soccer completed runs: {err}"))?;
        tx.commit()
            .map_err(|err| format!("commit soccer completed-run retention: {err}"))?;

        Ok(SoccerLearningPgCompletedRunRetentionPrune {
            deleted_runs,
            deleted_delta_rows,
        })
    }

    /// Mark old pgvector soccer moment rows as deleted without removing the raw
    /// rows. Retrieval paths filter `deleted_at is null`, while operators can
    /// inspect or hard-purge these rows later if needed.
    pub fn soft_delete_old_moment_vectors(
        &mut self,
        experiment_id: Option<&str>,
    ) -> Result<SoccerLearningPgMomentVectorRetentionPrune, String> {
        self.soft_delete_moment_vectors_older_than_days(
            experiment_id,
            SOCCER_MOMENT_VECTOR_SOFT_DELETE_AFTER_DAYS,
        )
    }

    pub fn soft_delete_moment_vectors_older_than_days(
        &mut self,
        experiment_id: Option<&str>,
        older_than_days: i64,
    ) -> Result<SoccerLearningPgMomentVectorRetentionPrune, String> {
        self.ensure_connected()?;
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer moment-vector retention transaction: {err}"))?;
        let prune =
            soft_delete_old_moment_vectors_in_transaction(&mut tx, experiment_id, older_than_days)?;
        tx.commit()
            .map_err(|err| format!("commit soccer moment-vector retention: {err}"))?;
        Ok(prune)
    }

    /// Roll one completed run's both-teams pass metrics into the per-commit learning-progress
    /// row in `des_soccer_learning_pass_metrics`, keyed by git commit. Each run from the same
    /// commit accumulates, so the table is a running series of pass-quality metrics over
    /// commits: completion rate (`passes_completed / passes_attempted`), average yards gained
    /// per pass (`completed_pass_gain_yards / passes_completed`), consecutive-pass chains and
    /// their total gained yards, chains that ended in a net loss, and shots on target that came
    /// off at least one pass. Yard sums are stored as micros (integer) like the other tables.
    pub fn upsert_pass_learning_metrics(
        &mut self,
        git_commit: &str,
        runs: u64,
        metrics: &SoccerPassLearningMetrics,
    ) -> Result<(), String> {
        self.ensure_connected()?;
        let git_commit = {
            let trimmed = git_commit.trim();
            if trimmed.is_empty() {
                "dev".to_string()
            } else {
                // Char-safe truncation to varchar(64) (avoid String::truncate's boundary panic).
                trimmed.chars().take(64).collect::<String>()
            }
        };
        let count = |value: u64| i64::try_from(value).unwrap_or(i64::MAX);
        let runs = count(runs);
        let passes_attempted = count(metrics.passes_attempted);
        let passes_completed = count(metrics.passes_completed);
        let completed_pass_gain_yards_micros =
            soccer_learning_to_micros(metrics.completed_pass_gain_yards);
        let pass_chains = count(metrics.pass_chains);
        let pass_chain_gain_yards_micros = soccer_learning_to_micros(metrics.pass_chain_gain_yards);
        let pass_chains_net_loss = count(metrics.pass_chains_net_loss);
        let shots_on_target = count(metrics.shots_on_target);
        let shots_after_pass = count(metrics.shots_after_pass);

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin pass-metrics upsert tx: {err}"))?;
        ensure_soccer_learning_pass_metrics_table(&mut tx)?;
        tx.execute(
            r#"
            insert into des_soccer_learning_pass_metrics as m (
              git_commit,
              runs,
              passes_attempted,
              passes_completed,
              completed_pass_gain_yards_micros,
              pass_chains,
              pass_chain_gain_yards_micros,
              pass_chains_net_loss,
              shots_on_target,
              shots_after_pass
            )
            values ($1, $10, $2, $3, $4, $5, $6, $7, $8, $9)
            on conflict (git_commit) do update set
              runs = m.runs + excluded.runs,
              passes_attempted = m.passes_attempted + excluded.passes_attempted,
              passes_completed = m.passes_completed + excluded.passes_completed,
              completed_pass_gain_yards_micros =
                m.completed_pass_gain_yards_micros + excluded.completed_pass_gain_yards_micros,
              pass_chains = m.pass_chains + excluded.pass_chains,
              pass_chain_gain_yards_micros =
                m.pass_chain_gain_yards_micros + excluded.pass_chain_gain_yards_micros,
              pass_chains_net_loss = m.pass_chains_net_loss + excluded.pass_chains_net_loss,
              shots_on_target = m.shots_on_target + excluded.shots_on_target,
              shots_after_pass = m.shots_after_pass + excluded.shots_after_pass,
              updated_at = now()
            "#,
            &[
                &git_commit,
                &passes_attempted,
                &passes_completed,
                &completed_pass_gain_yards_micros,
                &pass_chains,
                &pass_chain_gain_yards_micros,
                &pass_chains_net_loss,
                &shots_on_target,
                &shots_after_pass,
                &runs,
            ],
        )
        .map_err(|err| format!("upsert soccer learning pass metrics: {err}"))?;
        tx.commit()
            .map_err(|err| format!("commit pass-metrics upsert: {err}"))?;
        Ok(())
    }

    pub fn insert_set_play_training_artifact(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        base_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        status: &str,
        artifact: &SoccerSetPlayTrainingArtifact,
        elapsed_seconds: f64,
    ) -> Result<(String, String), String> {
        self.ensure_connected()?;
        let policies = SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries_with_targets(
                artifact.options.clone(),
                &artifact.home_entries,
                &artifact.home_target_entries,
            )?,
            away: SoccerQPolicy::from_entries_with_targets(
                artifact.options.clone(),
                &artifact.away_entries,
                &artifact.away_target_entries,
            )?,
        };
        let config_json = serde_json::to_value(&artifact.config)
            .map_err(|err| format!("serialize set-play config: {err}"))?;
        let options_json = json!({
            "home": &artifact.options,
            "away": &artifact.options,
        });
        let lineage = base_policy_version_id
            .map(|id| json!([id]))
            .unwrap_or_else(|| json!([]));
        let neural = json!({
            "enabled": artifact.learning.neural_learning_enabled,
            "backend": artifact.learning.neural_learning_backend,
            "trainingSteps": artifact.learning.neural_learning_training_steps,
            "samples": artifact.learning.neural_learning_samples,
            "pendingBatches": artifact.learning.neural_learning_pending_batches,
            "droppedBatches": artifact.learning.neural_learning_dropped_batches,
            "replaySamples": artifact.learning.neural_learning_replay_samples,
            "replayCapacity": artifact.learning.neural_learning_replay_capacity,
            "parameterCount": artifact.learning.neural_learning_parameter_count,
            "targetClip": artifact.learning.neural_learning_target_clip,
            "lastLoss": artifact.learning.neural_learning_last_loss,
            "averageLoss": artifact.learning.neural_learning_average_loss,
        });
        let base_metrics = json!({
            "fitness": artifact.goal_rate,
            "kind": "set-play-restart-training",
            "restart": &artifact.restart,
            "restarts": &artifact.restarts,
            "team": &artifact.team,
            "spot": &artifact.spot,
            "durationSeconds": artifact.duration_seconds,
            "episodes": artifact.episodes.len(),
            "goals": artifact.goals,
            "goalRate": artifact.goal_rate,
            "firstWindowGoalRate": artifact.first_window_goal_rate,
            "lastWindowGoalRate": artifact.last_window_goal_rate,
            "goalRateDelta": artifact.goal_rate_delta,
            "neural": neural,
        });
        let summary_json = json!({
            "kind": "set-play-restart-training",
            "restart": &artifact.restart,
            "restarts": &artifact.restarts,
            "team": &artifact.team,
            "spot": &artifact.spot,
            "durationSeconds": artifact.duration_seconds,
            "episodes": artifact.episodes.len(),
            "goals": artifact.goals,
            "goalRate": artifact.goal_rate,
            "firstWindowGoalRate": artifact.first_window_goal_rate,
            "lastWindowGoalRate": artifact.last_window_goal_rate,
            "goalRateDelta": artifact.goal_rate_delta,
        });
        let stats_json = json!({
            "learning": &artifact.learning,
            "neural": base_metrics["neural"].clone(),
            "episodes": &artifact.episodes,
        });
        let entry_count =
            checked_i32(policies.home.entries().len() + policies.away.entries().len());
        let target_entry_count = checked_i32(
            policies.home.target_entries().len() + policies.away.target_entries().len(),
        );
        let visit_count = checked_i64(policies.home.visit_count() + policies.away.visit_count());
        let fitness_micros = soccer_learning_to_micros(artifact.goal_rate);
        let score_home = checked_i32(if artifact.team == Team::Home {
            artifact.goals
        } else {
            0
        });
        let score_away = checked_i32(if artifact.team == Team::Away {
            artifact.goals
        } else {
            0
        });
        let home_goal_diff = if artifact.team == Team::Home {
            score_home
        } else {
            -score_away
        };
        let away_goal_diff = -home_goal_diff;
        let trained_team_scored = artifact.goals > 0;
        let (home_outcome, away_outcome) = match (artifact.team, trained_team_scored) {
            (Team::Home, true) => ("win", "loss"),
            (Team::Away, true) => ("loss", "win"),
            _ => ("draw", "draw"),
        };
        let trained_merge_weight = soccer_learning_to_micros(1.0 + artifact.goal_rate);
        let defending_merge_weight = soccer_learning_to_micros((1.0 - artifact.goal_rate) * 0.5);
        let (home_merge_weight_micros, away_merge_weight_micros) = match artifact.team {
            Team::Home => (trained_merge_weight, defending_merge_weight),
            Team::Away => (defending_merge_weight, trained_merge_weight),
        };
        let duration_ticks = checked_i64(
            artifact
                .episodes
                .iter()
                .map(|episode| episode.ticks)
                .sum::<u64>(),
        );
        let simulated_seconds = artifact
            .episodes
            .iter()
            .map(|episode| episode.simulated_seconds)
            .sum::<f64>();
        let simulated_seconds_micros = soccer_learning_to_micros(simulated_seconds);
        let elapsed_millis = (elapsed_seconds.max(0.0) * 1000.0).round() as i64;
        let transitions = checked_i32(
            artifact
                .episodes
                .iter()
                .map(|episode| episode.policy_updates)
                .sum::<u64>(),
        );

        self.ensure_policy_retention_schema_ready()?;
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer set-play training transaction: {err}"))?;
        ensure_soccer_learning_set_play_tables(&mut tx)?;
        let policy_version_id = Uuid::new_v4().to_string();
        let retention_kind = SOCCER_POLICY_RETENTION_BRANCH_TIP;
        let branch_key = soccer_policy_branch_key_for_insert(
            &mut tx,
            base_policy_version_id,
            &policy_version_id,
        )?;
        let metrics_for_status = soccer_policy_version_metrics_with_retention(
            base_metrics.clone(),
            &policy_version_id,
            &branch_key,
            retention_kind,
            soccer_policy_version_retains_full_entries(status),
        );
        let effective_status =
            soccer_policy_version_status_after_promotion_sample_floor(status, &metrics_for_status);
        let full_entries_retained = soccer_policy_version_retains_full_entries(effective_status);
        let metrics = soccer_policy_version_metrics_with_retention(
            base_metrics,
            &policy_version_id,
            &branch_key,
            retention_kind,
            full_entries_retained,
        );

        if effective_status == SOCCER_POLICY_STATUS_ACTIVE {
            // Serialize active-policy promotions PER EXPERIMENT so the
            // archive-old-active + insert-new-active pair is atomic against any
            // other promoter (a concurrent tournament run, the continuous learner,
            // a second replica). Without it, under READ COMMITTED two promoters can
            // each archive the other's not-yet-committed active and both insert,
            // leaving TWO rows at status='active'. This transaction-scoped lock is
            // auto-released on commit/rollback and only contends writers promoting
            // the SAME experiment (different experiments hash to different keys, so
            // the common single-writer path sees zero contention). Every promoter
            // funnels through this function, so locking here covers them all.
            tx.execute(
                "select pg_advisory_xact_lock(hashtext('des-soccer-policy-active-promote'), hashtext($1))",
                &[&experiment_id],
            )
            .map_err(|err| format!("acquire policy promotion advisory lock: {err}"))?;
            tx.execute(
                "select set_config('des_soccer.policy_active_promotion_in_progress', '1', true)",
                &[],
            )
            .map_err(|err| format!("mark policy promotion transaction: {err}"))?;
            tx.execute(
                r#"
                update des_soccer_learning_policy_versions
                set status = 'archived', updated_at = now()
                where experiment_id = $1::text::uuid and status = 'active'
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("archive old soccer policy versions: {err}"))?;
        }

        let inserted_policy_rows = tx
            .execute(
                r#"
                insert into des_soccer_learning_policy_versions
                  (
                    id,
                    experiment_id,
                    parent_policy_version_id,
                    generation,
                    version_label,
                    source_kind,
                    status,
                    options,
                    config,
                    lineage,
                    metrics,
                    entry_count,
                    target_entry_count,
                    visit_count,
                    fitness_micros,
                    branch_key,
                    retention_kind,
                    full_entries_retained
                  )
                values
                  (
                    $1::text::uuid,
                    $2::text::uuid,
                    $3::text::uuid,
                    $4,
                    $5,
                    'replay',
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15::text::uuid,
                    $16,
                    $17
                  )
                "#,
                &[
                    &policy_version_id,
                    &experiment_id,
                    &base_policy_version_id,
                    &generation,
                    &version_label,
                    &effective_status,
                    &options_json,
                    &config_json,
                    &lineage,
                    &metrics,
                    &entry_count,
                    &target_entry_count,
                    &visit_count,
                    &fitness_micros,
                    &branch_key,
                    &retention_kind,
                    &full_entries_retained,
                ],
            )
            .map_err(|err| format!("insert soccer set-play policy version: {err}"))?;
        if inserted_policy_rows != 1 {
            return Err(format!(
                "insert soccer set-play policy version inserted {inserted_policy_rows} rows for policy version {policy_version_id}"
            ));
        }

        let run_row = tx
            .query_one(
                r#"
                insert into des_soccer_learning_runs
                  (
                    experiment_id,
                    base_policy_version_id,
                    output_policy_version_id,
                    runner_id,
                    seed,
                    episode_index,
                    status,
                    score_home,
                    score_away,
                    home_goal_diff,
                    away_goal_diff,
                    home_outcome,
                    away_outcome,
                    home_merge_weight_micros,
                    away_merge_weight_micros,
                    fitness_micros,
                    duration_ticks,
                    simulated_seconds_micros,
                    elapsed_millis,
                    transitions,
                    summary,
                    stats
                  )
                values
                  (
                    $1::text::uuid,
                    $2::text::uuid,
                    $3::text::uuid,
                    $4,
                    $5,
                    0,
                    'completed',
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15,
                    $16,
                    $17,
                    $18,
                    $19,
                    $20
                  )
                returning id::text
                "#,
                &[
                    &experiment_id,
                    &base_policy_version_id,
                    &policy_version_id,
                    &runner_id,
                    &(artifact.config.seed as i64),
                    &score_home,
                    &score_away,
                    &home_goal_diff,
                    &away_goal_diff,
                    &home_outcome,
                    &away_outcome,
                    &home_merge_weight_micros,
                    &away_merge_weight_micros,
                    &fitness_micros,
                    &duration_ticks,
                    &simulated_seconds_micros,
                    &elapsed_millis,
                    &transitions,
                    &summary_json,
                    &stats_json,
                ],
            )
            .map_err(|err| format!("insert soccer set-play learning run: {err}"))?;
        let run_id: String = run_row.get(0);

        if full_entries_retained {
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Home,
                &policies.home,
                Some(&run_id),
            )?;
            insert_policy_entries_for_team(
                &mut tx,
                &policy_version_id,
                Team::Away,
                &policies.away,
                Some(&run_id),
            )?;
            let _ = prune_old_policy_entries_for_branch_tip(
                &mut tx,
                experiment_id,
                &branch_key,
                &policy_version_id,
            )?;
        }
        insert_normalized_set_play_training_records(
            &mut tx,
            &run_id,
            &policy_version_id,
            artifact,
        )?;

        tx.commit()
            .map_err(|err| format!("commit soccer set-play training transaction: {err}"))?;
        Ok((policy_version_id, run_id))
    }

    fn load_policy_entries(
        &mut self,
        policy_version_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        // Server-side trim: only entries with `visits >= min_visits` are returned.
        // `0` (the learner's resume) loads everything (the predicate is always true).
        // The inference-only live server passes `> 0` to skip the noisy single-visit
        // tail, so it transfers + holds the well-learned core (+ the net) instead of
        // the full multi-million-row policy. See `load_latest_active_policy_with_min_visits`.
        min_visits: i32,
    ) -> Result<SoccerTeamQPolicies, String> {
        let max_policy_entries = soccer_pg_resume_max_policy_entries();
        if max_policy_entries == Some(0) {
            println!(
                "soccer-learning-pg: skipping tabular policy entries for policy_version={} because {}=0; neural snapshot/config still load",
                policy_version_id, SOCCER_PG_RESUME_MAX_POLICY_ENTRIES_ENV
            );
            return Ok(SoccerTeamQPolicies {
                home: SoccerQPolicy::new(home_options),
                away: SoccerQPolicy::new(away_options),
            });
        }
        let mut home_entries = Vec::new();
        let mut away_entries = Vec::new();
        let mut home_targets = Vec::new();
        let mut away_targets = Vec::new();
        let mut loaded_entries = 0usize;

        // Keyset (seek) pagination over the FULL unique key — paging on only
        // (team, entry_kind, state_hash, action) would skip rows at a page boundary because
        // target entries can share that prefix across distinct target cells. There is no
        // total LIMIT: the entire policy must load to resume correctly; paging only bounds
        // how much is buffered in client memory per round-trip (vs the prior load-everything
        // -at-once query). The empty/MIN seed makes the first page's `> (...)` match all rows.
        let mut cursor_team = String::new();
        let mut cursor_kind = String::new();
        let mut cursor_hash = String::new();
        let mut cursor_action = String::new();
        let mut cursor_fine = i32::MIN;
        let mut cursor_tactical = i32::MIN;
        let mut cursor_macro = i32::MIN;
        let mut cursor_root = i32::MIN;
        let mut cursor_receiver = i32::MIN;
        loop {
            let rows = self
                .client
                .query(
                    r#"
                    select
                      team,
                      entry_kind,
                      state_key,
                      action,
                      target_fine_cell_id,
                      target_tactical_cell_id,
                      target_macro_cell_id,
                      target_root_cell_id,
                      value_micros,
                      visits,
                      state_hash,
                      receiver_descriptor
                    from des_soccer_learning_policy_entries
                    where policy_version_id = $1::text::uuid
                      and (team, entry_kind, state_hash, action,
                           target_fine_cell_id, target_tactical_cell_id,
                           target_macro_cell_id, target_root_cell_id,
                           receiver_descriptor)
                        > ($2::text, $3::text, $4::text, $5::text,
                           $6::int, $7::int, $8::int, $9::int, $12::int)
                      and visits >= $11::int
                    order by team, entry_kind, state_hash, action,
                             target_fine_cell_id, target_tactical_cell_id,
                             target_macro_cell_id, target_root_cell_id,
                             receiver_descriptor
                    limit $10
                    "#,
                    &[
                        &policy_version_id,
                        &cursor_team,
                        &cursor_kind,
                        &cursor_hash,
                        &cursor_action,
                        &cursor_fine,
                        &cursor_tactical,
                        &cursor_macro,
                        &cursor_root,
                        &SOCCER_POLICY_ENTRY_PAGE_SIZE,
                        &min_visits,
                        &cursor_receiver,
                    ],
                )
                .map_err(|err| format!("select soccer policy entries page: {err}"))?;
            let page_len = rows.len();
            // Advance the cursor to the last row of this page (rows are ordered by the key).
            if let Some(last) = rows.last() {
                cursor_team = last.get(0);
                cursor_kind = last.get(1);
                cursor_action = last.get(3);
                cursor_fine = last.get(4);
                cursor_tactical = last.get(5);
                cursor_macro = last.get(6);
                cursor_root = last.get(7);
                cursor_hash = last.get(10);
                cursor_receiver = last.get(11);
            }

            for row in &rows {
                if max_policy_entries.is_some_and(|max_entries| loaded_entries >= max_entries) {
                    break;
                }
                let team: String = row.get(0);
                let entry_kind: String = row.get(1);
                let state_key_json: Value = row.get(2);
                let state: SoccerQStateKey = serde_json::from_value(state_key_json)
                    .map_err(|err| format!("decode soccer policy state key: {err}"))?;
                let action: String = row.get(3);
                let target_fine_cell_id: i32 = row.get(4);
                let target_tactical_cell_id: i32 = row.get(5);
                let target_macro_cell_id: i32 = row.get(6);
                let target_root_cell_id: i32 = row.get(7);
                let value_micros: i64 = row.get(8);
                let visits_i32: i32 = row.get(9);
                let visits = visits_i32.max(0) as u32;
                let value = soccer_learning_from_micros(value_micros);
                let receiver_descriptor: i32 = row.get(11);
                match (team.as_str(), entry_kind.as_str()) {
                    ("home", "action") => home_entries.push(SoccerQEntry {
                        state,
                        action,
                        value,
                        visits,
                    }),
                    ("away", "action") => away_entries.push(SoccerQEntry {
                        state,
                        action,
                        value,
                        visits,
                    }),
                    ("home", "target") => home_targets.push(SoccerQTargetEntry {
                        state,
                        action,
                        target_fine_cell_id: target_fine_cell_id.max(0) as usize,
                        target_tactical_cell_id: target_tactical_cell_id.max(0) as usize,
                        target_macro_cell_id: target_macro_cell_id.max(0) as usize,
                        target_root_cell_id: target_root_cell_id.max(0) as usize,
                        receiver_descriptor,
                        value,
                        visits,
                    }),
                    ("away", "target") => away_targets.push(SoccerQTargetEntry {
                        state,
                        action,
                        target_fine_cell_id: target_fine_cell_id.max(0) as usize,
                        target_tactical_cell_id: target_tactical_cell_id.max(0) as usize,
                        target_macro_cell_id: target_macro_cell_id.max(0) as usize,
                        target_root_cell_id: target_root_cell_id.max(0) as usize,
                        receiver_descriptor,
                        value,
                        visits,
                    }),
                    _ => {}
                }
                loaded_entries = loaded_entries.saturating_add(1);
            }

            if max_policy_entries.is_some_and(|max_entries| loaded_entries >= max_entries) {
                println!(
                    "soccer-learning-pg: capped tabular policy entry load policy_version={} loaded_entries={} {}={}",
                    policy_version_id,
                    loaded_entries,
                    SOCCER_PG_RESUME_MAX_POLICY_ENTRIES_ENV,
                    loaded_entries
                );
                break;
            }

            // A short page means the last row of the version has been read.
            if page_len < SOCCER_POLICY_ENTRY_PAGE_SIZE as usize {
                break;
            }
        }

        // A policy version with zero entries loads as a blank policy and would
        // silently restart learning from scratch. This can legitimately happen
        // for a pruned/archived version, but never for the active version a run
        // resumes from — so surface it rather than collapsing quietly.
        if home_entries.is_empty()
            && away_entries.is_empty()
            && home_targets.is_empty()
            && away_targets.is_empty()
        {
            eprintln!(
                "soccer-learning-pg: loaded policy version {policy_version_id} with zero entries; \
                 resuming from this version would start learning from a blank policy"
            );
        }

        Ok(SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries_with_targets(
                home_options,
                &home_entries,
                &home_targets,
            )?,
            away: SoccerQPolicy::from_entries_with_targets(
                away_options,
                &away_entries,
                &away_targets,
            )?,
        })
    }
}

fn insert_completed_run_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    experiment_id: &str,
    runner_id: &str,
    base_policy_version_id: Option<&str>,
    output_policy_version_id: Option<&str>,
    game: &SoccerLearningCompletedGame,
) -> Result<String, String> {
    let summary_json =
        serde_json::to_value(&game.summary).map_err(|err| format!("serialize summary: {err}"))?;
    let stats_json = serde_json::to_value(&game.summary.stats)
        .map_err(|err| format!("serialize stats: {err}"))?;
    let seed = checked_i64(game.seed);
    let episode_index = checked_i32(game.episode);
    let score_home = checked_i32(game.summary.score_home);
    let score_away = checked_i32(game.summary.score_away);
    let home_goal_diff = game.score.home.goal_diff;
    let away_goal_diff = game.score.away.goal_diff;
    let home_outcome = game.score.home.outcome.as_str();
    let away_outcome = game.score.away.outcome.as_str();
    let home_merge_weight_micros = game.score.home.merge_weight_micros;
    let away_merge_weight_micros = game.score.away.merge_weight_micros;
    let duration_ticks = checked_i64(game.summary.ticks);
    let simulated_seconds_micros = soccer_learning_to_micros(game.summary.simulated_seconds);
    let elapsed_millis = (game.elapsed_seconds * 1000.0).round().max(0.0) as i64;
    let transitions = checked_i32(game.episode_summary.transitions);
    let row = tx
        .query_one(
            r#"
            insert into des_soccer_learning_runs
              (
                experiment_id,
                base_policy_version_id,
                output_policy_version_id,
                runner_id,
                seed,
                episode_index,
                status,
                score_home,
                score_away,
                home_goal_diff,
                away_goal_diff,
                home_outcome,
                away_outcome,
                home_merge_weight_micros,
                away_merge_weight_micros,
                fitness_micros,
                duration_ticks,
                simulated_seconds_micros,
                elapsed_millis,
                transitions,
                summary,
                stats
              )
            values
              (
                $1::text::uuid,
                $2::text::uuid,
                $3::text::uuid,
                $4,
                $5,
                $6,
                'completed',
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16,
                $17,
                $18,
                $19,
                $20,
                $21
              )
            returning id::text
            "#,
            &[
                &experiment_id,
                &base_policy_version_id,
                &output_policy_version_id,
                &runner_id,
                &seed,
                &episode_index,
                &score_home,
                &score_away,
                &home_goal_diff,
                &away_goal_diff,
                &home_outcome,
                &away_outcome,
                &home_merge_weight_micros,
                &away_merge_weight_micros,
                &game.score.match_fitness_micros,
                &duration_ticks,
                &simulated_seconds_micros,
                &elapsed_millis,
                &transitions,
                &summary_json,
                &stats_json,
            ],
        )
        .map_err(|err| format!("insert soccer learning run: {err}"))?;
    let run_id: String = row.get(0);

    if soccer_learning_pg_persist_run_deltas_enabled() {
        insert_run_delta_rows(tx, &run_id, &game.delta.entries)?;
    }

    Ok(run_id)
}

#[derive(Clone, Debug)]
struct SoccerCompletedRunHeaderInsert<'a> {
    run_id: String,
    base_policy_version_id: Option<&'a str>,
    output_policy_version_id: Option<&'a str>,
    seed: i64,
    episode_index: i32,
    score_home: i32,
    score_away: i32,
    home_goal_diff: i32,
    away_goal_diff: i32,
    home_outcome: &'static str,
    away_outcome: &'static str,
    home_merge_weight_micros: i64,
    away_merge_weight_micros: i64,
    fitness_micros: i64,
    duration_ticks: i64,
    simulated_seconds_micros: i64,
    elapsed_millis: i64,
    transitions: i32,
    summary_json: Value,
    stats_json: Value,
}

fn completed_run_header_insert<'a>(
    run: &SoccerLearningPgCompletedRunInsert<'a>,
) -> Result<SoccerCompletedRunHeaderInsert<'a>, String> {
    let game = run.game;
    let summary_json =
        serde_json::to_value(&game.summary).map_err(|err| format!("serialize summary: {err}"))?;
    let stats_json = serde_json::to_value(&game.summary.stats)
        .map_err(|err| format!("serialize stats: {err}"))?;
    Ok(SoccerCompletedRunHeaderInsert {
        run_id: Uuid::new_v4().to_string(),
        base_policy_version_id: run.base_policy_version_id,
        output_policy_version_id: run.output_policy_version_id,
        seed: checked_i64(game.seed),
        episode_index: checked_i32(game.episode),
        score_home: checked_i32(game.summary.score_home),
        score_away: checked_i32(game.summary.score_away),
        home_goal_diff: game.score.home.goal_diff,
        away_goal_diff: game.score.away.goal_diff,
        home_outcome: game.score.home.outcome.as_str(),
        away_outcome: game.score.away.outcome.as_str(),
        home_merge_weight_micros: game.score.home.merge_weight_micros,
        away_merge_weight_micros: game.score.away.merge_weight_micros,
        fitness_micros: game.score.match_fitness_micros,
        duration_ticks: checked_i64(game.summary.ticks),
        simulated_seconds_micros: soccer_learning_to_micros(game.summary.simulated_seconds),
        elapsed_millis: (game.elapsed_seconds * 1000.0).round().max(0.0) as i64,
        transitions: checked_i32(game.episode_summary.transitions),
        summary_json,
        stats_json,
    })
}

fn insert_completed_run_headers_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    experiment_id: &str,
    runner_id: &str,
    runs: &[SoccerLearningPgCompletedRunInsert<'_>],
) -> Result<Vec<String>, String> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }

    let batch_rows = runs
        .iter()
        .map(completed_run_header_insert)
        .collect::<Result<Vec<_>, _>>()?;
    let sql_prefix = r#"
        insert into des_soccer_learning_runs
          (
            id,
            experiment_id,
            base_policy_version_id,
            output_policy_version_id,
            runner_id,
            seed,
            episode_index,
            status,
            score_home,
            score_away,
            home_goal_diff,
            away_goal_diff,
            home_outcome,
            away_outcome,
            home_merge_weight_micros,
            away_merge_weight_micros,
            fitness_micros,
            duration_ticks,
            simulated_seconds_micros,
            elapsed_millis,
            transitions,
            summary,
            stats
          )
        values
        "#;
    let mut sql = postgres_insert_sql_buffer(
        sql_prefix,
        batch_rows.len(),
        SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT,
    );
    let mut params: Vec<&(dyn ToSql + Sync)> =
        Vec::with_capacity(batch_rows.len() * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT);
    for (idx, row) in batch_rows.iter().enumerate() {
        if idx > 0 {
            sql.push_str(", ");
        }
        append_completed_run_header_value_tuple(
            &mut sql,
            idx * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT + 1,
        );
        params.push(&row.run_id);
        params.push(&experiment_id);
        params.push(&row.base_policy_version_id);
        params.push(&row.output_policy_version_id);
        params.push(&runner_id);
        params.push(&row.seed);
        params.push(&row.episode_index);
        params.push(&row.score_home);
        params.push(&row.score_away);
        params.push(&row.home_goal_diff);
        params.push(&row.away_goal_diff);
        params.push(&row.home_outcome);
        params.push(&row.away_outcome);
        params.push(&row.home_merge_weight_micros);
        params.push(&row.away_merge_weight_micros);
        params.push(&row.fitness_micros);
        params.push(&row.duration_ticks);
        params.push(&row.simulated_seconds_micros);
        params.push(&row.elapsed_millis);
        params.push(&row.transitions);
        params.push(&row.summary_json);
        params.push(&row.stats_json);
    }
    let inserted = tx
        .execute(&sql, &params)
        .map_err(|err| format!("insert soccer learning run header batch: {err}"))?;
    if inserted as usize != batch_rows.len() {
        return Err(format!(
            "insert soccer learning run header batch inserted {inserted} rows for {} inputs",
            batch_rows.len()
        ));
    }
    Ok(batch_rows.into_iter().map(|row| row.run_id).collect())
}

#[derive(Clone, Copy, Debug)]
struct SoccerRunDeltaBatchEntryInsert<'a> {
    run_index: usize,
    delta: &'a SoccerLearningPolicyDeltaEntry,
    team: &'static str,
    entry_kind: &'static str,
    visit_delta: i32,
}

fn soccer_run_delta_batch_entry_insert(
    run_index: usize,
    delta: &SoccerLearningPolicyDeltaEntry,
) -> SoccerRunDeltaBatchEntryInsert<'_> {
    SoccerRunDeltaBatchEntryInsert {
        run_index,
        delta,
        team: soccer_team_label(delta.team),
        entry_kind: delta.entry_kind.as_str(),
        visit_delta: checked_i32(delta.visit_delta),
    }
}

fn insert_run_delta_rows(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    rows: &[SoccerLearningPolicyDeltaEntry],
) -> Result<(), String> {
    let run_ids = [run_id.to_string()];
    let mut batch_rows = Vec::with_capacity(SOCCER_RUN_DELTA_INSERT_BATCH_SIZE);
    for delta in rows {
        batch_rows.push(soccer_run_delta_batch_entry_insert(0, delta));
        if batch_rows.len() == SOCCER_RUN_DELTA_INSERT_BATCH_SIZE {
            insert_run_delta_batch_rows(tx, &run_ids, &batch_rows)?;
            batch_rows.clear();
        }
    }
    if !batch_rows.is_empty() {
        insert_run_delta_batch_rows(tx, &run_ids, &batch_rows)?;
    }
    Ok(())
}

fn insert_completed_run_delta_rows_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    run_ids: &[String],
    runs: &[SoccerLearningPgCompletedRunInsert<'_>],
) -> Result<(), String> {
    if run_ids.len() != runs.len() {
        return Err(format!(
            "insert soccer learning run delta batch got {} run ids for {} runs",
            run_ids.len(),
            runs.len()
        ));
    }
    let mut batch_rows = Vec::with_capacity(SOCCER_RUN_DELTA_INSERT_BATCH_SIZE);
    for (run_index, run) in runs.iter().enumerate() {
        for delta in &run.game.delta.entries {
            batch_rows.push(soccer_run_delta_batch_entry_insert(run_index, delta));
            if batch_rows.len() == SOCCER_RUN_DELTA_INSERT_BATCH_SIZE {
                insert_run_delta_batch_rows(tx, run_ids, &batch_rows)?;
                batch_rows.clear();
            }
        }
    }
    if !batch_rows.is_empty() {
        insert_run_delta_batch_rows(tx, run_ids, &batch_rows)?;
    }
    Ok(())
}

fn insert_run_delta_batch_rows(
    tx: &mut postgres::Transaction<'_>,
    run_ids: &[String],
    rows: &[SoccerRunDeltaBatchEntryInsert<'_>],
) -> Result<(), String> {
    for chunk in rows.chunks(SOCCER_RUN_DELTA_INSERT_BATCH_SIZE) {
        let sql_prefix = r#"
            insert into des_soccer_learning_run_deltas
              (
                run_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                target_fine_cell_id,
                target_tactical_cell_id,
                target_macro_cell_id,
                target_root_cell_id,
                receiver_descriptor,
                before_value_micros,
                after_value_micros,
                value_delta_micros,
                visit_delta,
                merge_weight_micros,
                effective_visit_micros
              )
            values
            "#;
        let mut sql =
            postgres_insert_sql_buffer(sql_prefix, chunk.len(), SOCCER_RUN_DELTA_PARAMETER_COUNT);
        let mut params: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(chunk.len() * SOCCER_RUN_DELTA_PARAMETER_COUNT);
        for (idx, batch_row) in chunk.iter().enumerate() {
            let delta = batch_row.delta;
            let run_id = run_ids.get(batch_row.run_index).ok_or_else(|| {
                format!(
                    "insert soccer learning run delta batch has row for missing run index {}",
                    batch_row.run_index
                )
            })?;
            if idx > 0 {
                sql.push_str(", ");
            }
            append_run_delta_value_tuple(&mut sql, idx * SOCCER_RUN_DELTA_PARAMETER_COUNT + 1);
            params.push(run_id);
            params.push(&batch_row.team);
            params.push(&batch_row.entry_kind);
            params.push(&delta.state_hash);
            params.push(&delta.state_json);
            params.push(&delta.action);
            params.push(&delta.target_fine_cell_id);
            params.push(&delta.target_tactical_cell_id);
            params.push(&delta.target_macro_cell_id);
            params.push(&delta.target_root_cell_id);
            params.push(&delta.receiver_descriptor);
            params.push(&delta.before_value_micros);
            params.push(&delta.after_value_micros);
            params.push(&delta.value_delta_micros);
            params.push(&batch_row.visit_delta);
            params.push(&delta.merge_weight_micros);
            params.push(&delta.effective_visit_micros);
        }
        // Idempotent batch insert: a retried run or an in-batch state_hash
        // collision must not abort the whole transaction. The conflict target
        // matches des_soccer_learning_run_deltas_key_uq.
        sql.push_str(SOCCER_RUN_DELTA_ON_CONFLICT_CLAUSE);
        tx.execute(&sql, &params)
            .map_err(|err| format!("insert soccer learning run delta batch: {err}"))?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SoccerPolicyActionEntryInsert {
    state_json: Value,
    state_hash: String,
    action: String,
    value_micros: i64,
    visits: i32,
}

#[derive(Clone, Debug)]
struct SoccerPolicyTargetEntryInsert {
    state_json: Value,
    state_hash: String,
    action: String,
    target_fine_cell_id: i32,
    target_tactical_cell_id: i32,
    target_macro_cell_id: i32,
    target_root_cell_id: i32,
    receiver_descriptor: i32,
    value_micros: i64,
    visits: i32,
}

fn policy_entry_insert_chunk_is_full(row_count: usize) -> bool {
    row_count >= SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE
}

fn insert_policy_entries_for_team(
    tx: &mut postgres::Transaction<'_>,
    policy_version_id: &str,
    team: Team,
    policy: &SoccerQPolicy,
    source_run_id: Option<&str>,
) -> Result<(), String> {
    let team_label = soccer_team_label(team);
    let mut action_rows = Vec::with_capacity(SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE);
    for entry in policy.entries() {
        let state_json = serde_json::to_value(&entry.state)
            .map_err(|err| format!("serialize soccer action state key: {err}"))?;
        action_rows.push(SoccerPolicyActionEntryInsert {
            state_hash: state_hash(&state_json),
            state_json,
            action: entry.action,
            value_micros: soccer_learning_to_micros(entry.value),
            visits: checked_i32(entry.visits),
        });
        if policy_entry_insert_chunk_is_full(action_rows.len()) {
            insert_policy_action_entry_rows(
                tx,
                policy_version_id,
                &team_label,
                source_run_id,
                &action_rows,
            )?;
            action_rows.clear();
        }
    }
    if !action_rows.is_empty() {
        insert_policy_action_entry_rows(
            tx,
            policy_version_id,
            &team_label,
            source_run_id,
            &action_rows,
        )?;
    }

    let mut target_rows = Vec::with_capacity(SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE);
    for entry in policy.target_entries() {
        let state_json = serde_json::to_value(&entry.state)
            .map_err(|err| format!("serialize soccer target state key: {err}"))?;
        target_rows.push(SoccerPolicyTargetEntryInsert {
            state_hash: state_hash(&state_json),
            state_json,
            action: entry.action,
            target_fine_cell_id: checked_i32(entry.target_fine_cell_id),
            target_tactical_cell_id: checked_i32(entry.target_tactical_cell_id),
            target_macro_cell_id: checked_i32(entry.target_macro_cell_id),
            target_root_cell_id: checked_i32(entry.target_root_cell_id),
            receiver_descriptor: entry.receiver_descriptor,
            value_micros: soccer_learning_to_micros(entry.value),
            visits: checked_i32(entry.visits),
        });
        if policy_entry_insert_chunk_is_full(target_rows.len()) {
            insert_policy_target_entry_rows(
                tx,
                policy_version_id,
                &team_label,
                source_run_id,
                &target_rows,
            )?;
            target_rows.clear();
        }
    }
    if !target_rows.is_empty() {
        insert_policy_target_entry_rows(
            tx,
            policy_version_id,
            &team_label,
            source_run_id,
            &target_rows,
        )?;
    }
    Ok(())
}

fn insert_policy_action_entry_rows(
    tx: &mut postgres::Transaction<'_>,
    policy_version_id: &str,
    team_label: &str,
    source_run_id: Option<&str>,
    rows: &[SoccerPolicyActionEntryInsert],
) -> Result<(), String> {
    let entry_kind = SoccerLearningPolicyEntryKind::Action.as_str();
    for chunk in rows.chunks(SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE) {
        let sql_prefix = r#"
            insert into des_soccer_learning_policy_entries
              (
                policy_version_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                value_micros,
                visits,
                source_run_id
              )
            values
            "#;
        let mut sql = postgres_insert_sql_buffer(
            sql_prefix,
            chunk.len(),
            SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT,
        );
        let mut params: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(chunk.len() * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT);
        for (idx, row) in chunk.iter().enumerate() {
            if idx > 0 {
                sql.push_str(", ");
            }
            append_policy_entry_value_tuple(
                &mut sql,
                idx * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT + 1,
                false,
            );
            params.push(&policy_version_id);
            params.push(&team_label);
            params.push(&entry_kind);
            params.push(&row.state_hash);
            params.push(&row.state_json);
            params.push(&row.action);
            params.push(&row.value_micros);
            params.push(&row.visits);
            params.push(&source_run_id);
        }
        // Idempotent batch insert: tolerate in-batch state_hash collisions and
        // re-inserts under the same policy_version_id. Conflict target matches
        // des_soccer_learning_policy_entries_key_uq.
        sql.push_str(SOCCER_POLICY_ENTRY_ON_CONFLICT_CLAUSE);
        tx.execute(&sql, &params)
            .map_err(|err| format!("insert soccer policy action entry batch: {err}"))?;
    }
    Ok(())
}

fn insert_policy_target_entry_rows(
    tx: &mut postgres::Transaction<'_>,
    policy_version_id: &str,
    team_label: &str,
    source_run_id: Option<&str>,
    rows: &[SoccerPolicyTargetEntryInsert],
) -> Result<(), String> {
    let entry_kind = SoccerLearningPolicyEntryKind::Target.as_str();
    for chunk in rows.chunks(SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE) {
        let sql_prefix = r#"
            insert into des_soccer_learning_policy_entries
              (
                policy_version_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                target_fine_cell_id,
                target_tactical_cell_id,
                target_macro_cell_id,
                target_root_cell_id,
                receiver_descriptor,
                value_micros,
                visits,
                source_run_id
              )
            values
            "#;
        let mut sql = postgres_insert_sql_buffer(
            sql_prefix,
            chunk.len(),
            SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT,
        );
        let mut params: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(chunk.len() * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT);
        for (idx, row) in chunk.iter().enumerate() {
            if idx > 0 {
                sql.push_str(", ");
            }
            append_policy_entry_value_tuple(
                &mut sql,
                idx * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT + 1,
                true,
            );
            params.push(&policy_version_id);
            params.push(&team_label);
            params.push(&entry_kind);
            params.push(&row.state_hash);
            params.push(&row.state_json);
            params.push(&row.action);
            params.push(&row.target_fine_cell_id);
            params.push(&row.target_tactical_cell_id);
            params.push(&row.target_macro_cell_id);
            params.push(&row.target_root_cell_id);
            params.push(&row.receiver_descriptor);
            params.push(&row.value_micros);
            params.push(&row.visits);
            params.push(&source_run_id);
        }
        // Idempotent batch insert: tolerate in-batch state_hash collisions and
        // re-inserts under the same policy_version_id. Conflict target matches
        // des_soccer_learning_policy_entries_key_uq.
        sql.push_str(SOCCER_POLICY_ENTRY_ON_CONFLICT_CLAUSE);
        tx.execute(&sql, &params)
            .map_err(|err| format!("insert soccer policy target entry batch: {err}"))?;
    }
    Ok(())
}

fn append_completed_run_header_value_tuple(sql: &mut String, first_param: usize) {
    write!(
        sql,
        "(${}::text::uuid, ${}::text::uuid, ${}::text::uuid, ${}::text::uuid, ${}, ${}, ${}, 'completed', ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
        first_param,
        first_param + 1,
        first_param + 2,
        first_param + 3,
        first_param + 4,
        first_param + 5,
        first_param + 6,
        first_param + 7,
        first_param + 8,
        first_param + 9,
        first_param + 10,
        first_param + 11,
        first_param + 12,
        first_param + 13,
        first_param + 14,
        first_param + 15,
        first_param + 16,
        first_param + 17,
        first_param + 18,
        first_param + 19,
        first_param + 20,
        first_param + 21
    )
    .expect("write completed run header tuple");
}

fn append_run_delta_value_tuple(sql: &mut String, first_param: usize) {
    write!(
        sql,
        "(${}::text::uuid, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
        first_param,
        first_param + 1,
        first_param + 2,
        first_param + 3,
        first_param + 4,
        first_param + 5,
        first_param + 6,
        first_param + 7,
        first_param + 8,
        first_param + 9,
        first_param + 10,
        first_param + 11,
        first_param + 12,
        first_param + 13,
        first_param + 14,
        first_param + 15,
        first_param + 16
    )
    .expect("write run delta tuple");
}

fn append_policy_entry_value_tuple(
    sql: &mut String,
    first_param: usize,
    include_target_cells: bool,
) {
    if include_target_cells {
        write!(
            sql,
            "(${}::text::uuid, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}::text::uuid)",
            first_param,
            first_param + 1,
            first_param + 2,
            first_param + 3,
            first_param + 4,
            first_param + 5,
            first_param + 6,
            first_param + 7,
            first_param + 8,
            first_param + 9,
            first_param + 10,
            first_param + 11,
            first_param + 12,
            first_param + 13
        )
        .expect("write target policy entry tuple");
    } else {
        write!(
            sql,
            "(${}::text::uuid, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}::text::uuid)",
            first_param,
            first_param + 1,
            first_param + 2,
            first_param + 3,
            first_param + 4,
            first_param + 5,
            first_param + 6,
            first_param + 7,
            first_param + 8
        )
        .expect("write action policy entry tuple");
    }
}

fn ensure_soccer_learning_policy_retention_columns(
    tx: &mut postgres::Transaction<'_>,
) -> Result<(), String> {
    tx.batch_execute(
        r#"
        alter table des_soccer_learning_policy_versions
          add column if not exists branch_key uuid,
          add column if not exists retention_kind varchar(32) default 'branch_tip' not null,
          add column if not exists full_entries_retained boolean default true not null,
          add column if not exists full_entries_pruned_at timestamptz;

        with recursive policy_lineage as (
          select
            id,
            id as root_branch_key
          from des_soccer_learning_policy_versions
          where parent_policy_version_id is null
          union all
          select
            child.id,
            parent.root_branch_key
          from des_soccer_learning_policy_versions child
          join policy_lineage parent
            on child.parent_policy_version_id = parent.id
        )
        update des_soccer_learning_policy_versions versions
        set branch_key = policy_lineage.root_branch_key
        from policy_lineage
        where versions.id = policy_lineage.id
          and (
            versions.branch_key is null
            or (
              versions.branch_key = versions.id
              and versions.parent_policy_version_id is not null
            )
          );

        alter table des_soccer_learning_policy_versions
          alter column branch_key set not null;

        do $$
        begin
          if not exists (
            select 1
            from pg_constraint
            where conname = 'des_soccer_learning_policy_versions_retention_kind_chk'
          ) then
            alter table des_soccer_learning_policy_versions
              add constraint des_soccer_learning_policy_versions_retention_kind_chk
              check (retention_kind in ('branch_tip', 'retain_all', 'metadata_only'));
          end if;
        end $$;

        create index if not exists des_soccer_learning_policy_versions_branch_tip_idx
          on des_soccer_learning_policy_versions (
            experiment_id,
            branch_key,
            generation desc,
            updated_at desc
          )
          where full_entries_retained = true;
        create index if not exists des_soccer_learning_policy_versions_active_metadata_idx
          on des_soccer_learning_policy_versions (
            experiment_id,
            generation desc,
            updated_at desc,
            id desc
          )
          where status = 'active';
        "#,
    )
    .map_err(|err| format!("ensure soccer policy retention columns: {err}"))?;
    Ok(())
}

fn soccer_policy_branch_key_for_insert(
    tx: &mut postgres::Transaction<'_>,
    parent_policy_version_id: Option<&str>,
    policy_version_id: &str,
) -> Result<String, String> {
    if let Some(parent_policy_version_id) = parent_policy_version_id {
        if let Some(row) = tx
            .query_opt(
                r#"
                select branch_key::text
                from des_soccer_learning_policy_versions
                where id = $1::text::uuid
                limit 1
                "#,
                &[&parent_policy_version_id],
            )
            .map_err(|err| format!("select parent soccer policy branch key: {err}"))?
        {
            return Ok(row.get(0));
        }
    }
    Ok(policy_version_id.to_string())
}

fn soccer_completed_run_retention_limit(keep_latest_runs: usize) -> Option<i64> {
    if keep_latest_runs == 0 {
        None
    } else {
        Some(keep_latest_runs.min(i64::MAX as usize) as i64)
    }
}

fn prune_old_policy_entries_for_branch_tip(
    tx: &mut postgres::Transaction<'_>,
    experiment_id: &str,
    branch_key: &str,
    policy_version_id: &str,
) -> Result<u64, String> {
    tx.execute(
        r#"
        with pruned_versions as (
          update des_soccer_learning_policy_versions
          set
            full_entries_retained = false,
            full_entries_pruned_at = now(),
            updated_at = now(),
            metrics = jsonb_set(
              metrics,
              '{postgresRetention}',
              coalesce(metrics->'postgresRetention', '{}'::jsonb)
                || jsonb_build_object(
                  'fullEntriesRetained', false,
                  'prunedByPolicyVersionId', $3,
                  'retentionKind', 'branch_tip'
                ),
              true
            )
          where experiment_id = $1::text::uuid
            and branch_key = $2::text::uuid
            and id <> $3::text::uuid
            and full_entries_retained = true
          returning id
        )
        delete from des_soccer_learning_policy_entries entries
        using pruned_versions
        where entries.policy_version_id = pruned_versions.id
        "#,
        &[&experiment_id, &branch_key, &policy_version_id],
    )
    .map_err(|err| format!("prune old soccer policy branch entries: {err}"))
}

/// One retrieved neighbour from [`SoccerLearningPgStore::search_nearest_moments`].
#[derive(Clone, Debug)]
pub struct SoccerMomentNeighbor {
    pub team: String,
    pub action: String,
    pub reward: f64,
    /// Critic value of the neighbour moment (`None` if it was stored untrained).
    pub value: Option<f64>,
    pub tick: i64,
    /// Cosine distance to the query (0 = identical, 1 = orthogonal, 2 = opposite).
    pub distance: f64,
}

/// Serialize a float vector to pgvector's text input form `[v1,v2,...]`,
/// scrubbing non-finite components to `0` so the cast never fails.
fn pg_vector_text(values: &[f64]) -> String {
    let mut text = String::with_capacity(values.len() * 8 + 2);
    text.push('[');
    for (index, &value) in values.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        let value = if value.is_finite() { value } else { 0.0 };
        let _ = write!(text, "{value}");
    }
    text.push(']');
    text
}

/// Max moments per multi-row insert statement. Each row binds 6 params (plus the
/// 2 shared run/experiment ids), so 512 rows = 3074 params — well under
/// Postgres's 65535-parameter ceiling.
const MOMENT_EMBEDDING_INSERT_CHUNK_ROWS: usize = 512;

/// Build the `values (...),(...)` clause for a chunked moment-embedding insert.
/// `$1`/`$2` are the shared run_id/experiment_id reused by every row; each row's
/// 6 columns start at `$3 + 6*i`. Kept separate so the placeholder/offset
/// arithmetic is unit-tested without a live database.
fn moment_embedding_values_clause(rows: usize) -> String {
    let mut clause = String::new();
    for i in 0..rows {
        if i > 0 {
            clause.push(',');
        }
        let base = 3 + i * 6;
        let _ = write!(
            clause,
            "($1::text::uuid,$2::text::uuid,${},${},${},${},${},${}::text::vector)",
            base,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 5
        );
    }
    clause
}

/// Classify a (normalized) action label as attacking (`true`), defending
/// (`false`), or neutral (`None`) for the retrieval-to-LP signal.
fn soccer_moment_action_side(action: &str) -> Option<bool> {
    let action = action.to_ascii_lowercase();
    let defends = ["tackle", "defend", "clear", "block", "mark", "intercept"];
    let attacks = [
        "pass", "shoot", "shot", "dribble", "carry", "cross", "through", "run", "overlap",
    ];
    if defends.iter().any(|kind| action.contains(kind)) {
        Some(false)
    } else if attacks.iter().any(|kind| action.contains(kind)) {
        Some(true)
    } else {
        None
    }
}

/// Reduce retrieved neighbours into the LP's `(attack, defense)` embedding
/// signal: a similarity-weighted average of how *favourable* past moments like
/// this one were, split by whether the neighbour's action was attacking or
/// defending. Both outputs land in `[0, 1]`, matching the existing adversarial
/// embedding channel the formation LP already consumes — so RAG retrieval can
/// feed the same seam as the in-process moment index.
pub fn soccer_moment_neighbors_attack_defense(neighbors: &[SoccerMomentNeighbor]) -> (f64, f64) {
    let mut attack = (0.0_f64, 0.0_f64); // (weighted favourable sum, weight sum)
    let mut defense = (0.0_f64, 0.0_f64);
    for neighbor in neighbors {
        // Cosine distance ∈ [0,2]; similarity = 1 - distance, kept ≥ 0.
        let similarity = (1.0 - neighbor.distance).clamp(0.0, 1.0);
        if similarity <= f64::EPSILON {
            continue;
        }
        // Outcome: prefer the critic value, fall back to immediate reward;
        // squash to (-1,1) and keep only the favourable (positive) half.
        let favourable = neighbor.value.unwrap_or(neighbor.reward).tanh().max(0.0);
        match soccer_moment_action_side(&neighbor.action) {
            Some(true) => {
                attack.0 += similarity * favourable;
                attack.1 += similarity;
            }
            Some(false) => {
                defense.0 += similarity * favourable;
                defense.1 += similarity;
            }
            None => {}
        }
    }
    let resolve = |(sum, weight): (f64, f64)| {
        if weight > 0.0 {
            (sum / weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    (resolve(attack), resolve(defense))
}

/// Schema for persisted moment embeddings. Requires the `vector` extension —
/// `create extension if not exists vector` is a no-op when an admin has already
/// enabled it (the RDS-allowlisted path), and only needs elevated privilege the
/// first time. The `vector(N)` width is `SOCCER_MOMENT_EMBEDDING_DIM`.
/// Durable storage for nightly learning tournaments: one header per tournament,
/// every match, and each team's final evolved brain (neural snapshot + record).
fn ensure_soccer_tournament_tables(tx: &mut postgres::Transaction<'_>) -> Result<(), String> {
    tx.batch_execute(
        r#"
        create table if not exists des_soccer_tournaments (
          id bigserial primary key,
          experiment_id uuid not null,
          tournament_date text not null,
          seed bigint not null,
          learning_mode text not null,
          format jsonb not null,
          team_count integer not null,
          match_count integer not null default 0,
          matches_played integer not null default 0,
          champion_team_id integer,
          runner_up_team_id integer,
          third_place_team_id integer,
          wall_time_seconds double precision,
          status text not null default 'running',
          created_at timestamptz not null default now(),
          updated_at timestamptz not null default now(),
          finished_at timestamptz
        );
        create index if not exists des_soccer_tournaments_experiment_idx
          on des_soccer_tournaments (experiment_id, created_at desc);

        create table if not exists des_soccer_tournament_matches (
          id bigserial primary key,
          tournament_id bigint not null
            references des_soccer_tournaments(id) on delete cascade,
          match_index integer not null,
          stage text not null,
          home_team_id integer not null,
          away_team_id integer not null,
          home_goals integer not null,
          away_goals integer not null,
          shootout_winner_team_id integer,
          home_training_steps bigint not null,
          away_training_steps bigint not null,
          recorded_at timestamptz not null default now(),
          unique (tournament_id, match_index)
        );

        create table if not exists des_soccer_tournament_team_brains (
          id bigserial primary key,
          tournament_id bigint not null
            references des_soccer_tournaments(id) on delete cascade,
          team_id integer not null,
          team_name text not null,
          seed bigint not null,
          matches_learned integer not null,
          training_steps bigint not null,
          played integer not null,
          wins integer not null,
          draws integer not null,
          losses integer not null,
          goals_for integer not null,
          goals_against integer not null,
          neural_snapshot jsonb,
          genome jsonb,
          updated_at timestamptz not null default now(),
          unique (tournament_id, team_id)
        );
        -- Heritable tactical genome (added after the table shipped); idempotent so
        -- pre-existing prod tables gain the column without a separate migration.
        alter table des_soccer_tournament_team_brains
          add column if not exists genome jsonb;
        "#,
    )
    .map_err(|err| format!("ensure soccer tournament tables: {err}"))
}

/// One retrieved neighbour from [`SoccerLearningPgStore::search_nearest_config_moments`].
#[derive(Clone, Debug)]
pub struct SoccerConfigMomentNeighbor {
    pub team: String,
    pub role: String,
    pub action: String,
    pub reward: f64,
    /// Discounted n-step outcome return stored with the moment.
    pub nstep_return: f64,
    /// Critic value of the neighbour (`None` if stored untrained).
    pub value: Option<f64>,
    pub tick: i64,
    /// Raw canonical config features for an exact permutation re-score.
    pub features: Vec<f32>,
    /// Cosine distance to the query (0 = identical, 1 = orthogonal, 2 = opposite).
    pub distance: f64,
}

fn soccer_role_label(role: PlayerRole) -> &'static str {
    match role {
        PlayerRole::Goalkeeper => "goalkeeper",
        PlayerRole::Defender => "defender",
        PlayerRole::Midfielder => "midfielder",
        PlayerRole::Forward => "forward",
    }
}

/// Scrub non-finite feature components to 0 so the `real[]` cast never fails and
/// retrieval geometry stays well-defined.
fn sanitize_features(features: &[f32]) -> Vec<f32> {
    features
        .iter()
        .map(|&f| if f.is_finite() { f } else { 0.0 })
        .collect()
}

fn validate_config_moments(moments: &[SoccerConfigMomentInsert]) -> Result<(), String> {
    for moment in moments {
        if moment.embedding.len() != SOCCER_MOMENT_EMBEDDING_DIM {
            return Err(format!(
                "config moment embedding has {} dims, expected {SOCCER_MOMENT_EMBEDDING_DIM}",
                moment.embedding.len()
            ));
        }
        if moment.embedding_assigned.len() != SOCCER_MOMENT_EMBEDDING_DIM {
            return Err(format!(
                "config moment assigned embedding has {} dims, expected {SOCCER_MOMENT_EMBEDDING_DIM}",
                moment.embedding_assigned.len()
            ));
        }
        if moment.features.len() != CONFIG_FEATURE_DIM {
            return Err(format!(
                "config moment features have {} dims, expected {CONFIG_FEATURE_DIM}",
                moment.features.len()
            ));
        }
        if moment.features_assigned.len() != CONFIG_FEATURE_DIM {
            return Err(format!(
                "config moment assigned features have {} dims, expected {CONFIG_FEATURE_DIM}",
                moment.features_assigned.len()
            ));
        }
    }
    Ok(())
}

fn insert_config_moments_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    run_id: Option<&str>,
    experiment_id: Option<&str>,
    moments: &[SoccerConfigMomentInsert],
) -> Result<(), String> {
    // Chunked multi-row insert: 11 params per row plus the 2 shared run/experiment
    // ids. CONFIG_MOMENT_INSERT_CHUNK_ROWS keeps the total under Postgres's
    // 65535-parameter ceiling.
    for chunk in moments.chunks(CONFIG_MOMENT_INSERT_CHUNK_ROWS) {
        let teams: Vec<&'static str> = chunk.iter().map(|m| soccer_team_label(m.team)).collect();
        let ticks: Vec<i64> = chunk.iter().map(|m| checked_i64(m.tick)).collect();
        let roles: Vec<&'static str> = chunk.iter().map(|m| soccer_role_label(m.role)).collect();
        let rewards: Vec<i64> = chunk
            .iter()
            .map(|m| soccer_learning_to_micros(m.reward))
            .collect();
        let returns: Vec<i64> = chunk
            .iter()
            .map(|m| soccer_learning_to_micros(m.nstep_return))
            .collect();
        let values: Vec<Option<i64>> = chunk
            .iter()
            .map(|m| m.value.map(soccer_learning_to_micros))
            .collect();
        let embeddings: Vec<String> = chunk.iter().map(|m| pg_vector_text(&m.embedding)).collect();
        let features: Vec<Vec<f32>> = chunk
            .iter()
            .map(|m| sanitize_features(&m.features))
            .collect();
        let embeddings_assigned: Vec<String> = chunk
            .iter()
            .map(|m| pg_vector_text(&m.embedding_assigned))
            .collect();
        let features_assigned: Vec<Vec<f32>> = chunk
            .iter()
            .map(|m| sanitize_features(&m.features_assigned))
            .collect();
        let sql = format!(
            "insert into des_soccer_config_moments \
             (run_id, experiment_id, team, tick, role, action, reward_micros, \
              nstep_return_micros, value_micros, embedding, features, \
              embedding_assigned, features_assigned) \
             values {}",
            config_moment_values_clause(chunk.len())
        );
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&run_id, &experiment_id];
        for (i, moment) in chunk.iter().enumerate() {
            params.push(&teams[i]);
            params.push(&ticks[i]);
            params.push(&roles[i]);
            params.push(&moment.action);
            params.push(&rewards[i]);
            params.push(&returns[i]);
            params.push(&values[i]);
            params.push(&embeddings[i]);
            params.push(&features[i]);
            params.push(&embeddings_assigned[i]);
            params.push(&features_assigned[i]);
        }
        tx.execute(&sql, &params)
            .map_err(|err| format!("insert soccer config moments: {err}"))?;
    }
    Ok(())
}

/// Max config moments per multi-row insert. Each row binds 11 params plus the 2
/// shared run/experiment ids, so 256 rows = 2818 params — well under the ceiling.
const CONFIG_MOMENT_INSERT_CHUNK_ROWS: usize = 256;

/// Build the `values (...),(...)` clause for a chunked config-moment insert.
/// `$1`/`$2` are the shared run_id/experiment_id reused by every row; each row's
/// 11 columns (…, embedding, features, embedding_assigned, features_assigned)
/// start at `$3 + 11*i`. Kept separate so the placeholder/offset arithmetic is
/// unit-tested without a live database.
fn config_moment_values_clause(rows: usize) -> String {
    let mut clause = String::new();
    for i in 0..rows {
        if i > 0 {
            clause.push(',');
        }
        let base = 3 + i * 11;
        let _ = write!(
            clause,
            "($1::text::uuid,$2::text::uuid,${}::text,${}::bigint,${}::text,${}::text,\
             ${}::bigint,${}::bigint,${}::bigint,${}::text::vector,${}::real[],${}::text::vector,${}::real[])",
            base,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 5,
            base + 6,
            base + 7,
            base + 8,
            base + 9,
            base + 10
        );
    }
    clause
}

/// Multi-row VALUES clause for pass-outcome samples: `$1,$2` are the shared run/experiment ids,
/// then 3 params per row (features real[], completed bool, own_half bool).
fn pass_outcome_values_clause(rows: usize) -> String {
    let mut clause = String::new();
    for i in 0..rows {
        if i > 0 {
            clause.push(',');
        }
        let base = 3 + i * 3;
        let _ = write!(
            clause,
            "($1::text::uuid,$2::text::uuid,${},${},${})",
            base,
            base + 1,
            base + 2
        );
    }
    clause
}

/// 3 params/row + 2 shared ⇒ stay well under Postgres's 65535-parameter ceiling.
const PASS_OUTCOME_INSERT_CHUNK_ROWS: usize = 5000;

fn ensure_soccer_pass_outcome_tables(tx: &mut postgres::Transaction<'_>) -> Result<(), String> {
    tx.batch_execute(&format!(
        r#"
        create extension if not exists pgcrypto;
        create table if not exists des_soccer_pass_outcome_samples (
          id uuid primary key default gen_random_uuid(),
          run_id uuid,
          experiment_id uuid,
          features real[] not null,
          completed boolean not null,
          own_half boolean not null,
          created_at timestamptz not null default now(),
          deleted_at timestamptz
        );
        do $$
        begin
          if not exists (
            select 1
            from pg_constraint
            where conname = 'des_soccer_pass_outcome_features_len_chk'
              and conrelid = 'des_soccer_pass_outcome_samples'::regclass
          ) then
            begin
              alter table des_soccer_pass_outcome_samples
                add constraint des_soccer_pass_outcome_features_len_chk
                  check (array_length(features, 1) in ({legacy_dim}, {dim}));
            exception when duplicate_object then
              null;
            end;
          end if;
        end $$;
        create index if not exists des_soccer_pass_outcome_run_idx
          on des_soccer_pass_outcome_samples (run_id);
        create index if not exists des_soccer_pass_outcome_live_created_idx
          on des_soccer_pass_outcome_samples (experiment_id, created_at)
          where deleted_at is null;
        "#,
        dim = SOCCER_PASS_COMPLETION_FEATURE_DIM,
        legacy_dim = SOCCER_PASS_COMPLETION_FEATURE_DIM_V1
    ))
    .map_err(|err| format!("ensure soccer pass-outcome tables: {err}"))
}

fn insert_pass_outcome_samples_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    run_id: Option<&str>,
    experiment_id: Option<&str>,
    samples: &[SoccerPassOutcomeSample],
) -> Result<(), String> {
    for chunk in samples.chunks(PASS_OUTCOME_INSERT_CHUNK_ROWS) {
        let features: Vec<Vec<f32>> = chunk
            .iter()
            .map(|s| sanitize_features(&s.features))
            .collect();
        let completed: Vec<bool> = chunk.iter().map(|s| s.completed).collect();
        let own_half: Vec<bool> = chunk.iter().map(|s| s.own_half).collect();
        let sql = format!(
            "insert into des_soccer_pass_outcome_samples \
             (run_id, experiment_id, features, completed, own_half) values {}",
            pass_outcome_values_clause(chunk.len())
        );
        let mut params: Vec<&(dyn ToSql + Sync)> = vec![&run_id, &experiment_id];
        for i in 0..chunk.len() {
            params.push(&features[i]);
            params.push(&completed[i]);
            params.push(&own_half[i]);
        }
        tx.execute(sql.as_str(), &params)
            .map_err(|err| format!("insert soccer pass-outcome samples: {err}"))?;
    }
    Ok(())
}

fn soccer_moment_vector_retention_limit_days(days: i64) -> Option<i64> {
    (days > 0).then_some(days)
}

fn soft_delete_old_moment_vectors_in_transaction(
    tx: &mut postgres::Transaction<'_>,
    experiment_id: Option<&str>,
    older_than_days: i64,
) -> Result<SoccerLearningPgMomentVectorRetentionPrune, String> {
    let Some(older_than_days) = soccer_moment_vector_retention_limit_days(older_than_days) else {
        return Ok(SoccerLearningPgMomentVectorRetentionPrune::default());
    };
    ensure_soccer_moment_embedding_tables(tx)?;
    ensure_soccer_config_moment_tables(tx)?;
    let soft_deleted_moment_embeddings = tx
        .execute(
            SOCCER_MOMENT_EMBEDDING_SOFT_DELETE_SQL,
            &[&experiment_id, &older_than_days],
        )
        .map_err(|err| format!("soft-delete old soccer moment embeddings: {err}"))?;
    let soft_deleted_config_moments = tx
        .execute(
            SOCCER_CONFIG_MOMENT_SOFT_DELETE_SQL,
            &[&experiment_id, &older_than_days],
        )
        .map_err(|err| format!("soft-delete old soccer config moments: {err}"))?;
    Ok(SoccerLearningPgMomentVectorRetentionPrune {
        soft_deleted_moment_embeddings,
        soft_deleted_config_moments,
    })
}

/// Schema for persisted configuration moments — the retrieval corpus. Mirrors the
/// moment-embedding table but adds the role, the n-step outcome return, and the
/// raw `features real[]` (the canonical per-player floats kept for an exact
/// permutation re-score). `vector(dim)` width is [`SOCCER_MOMENT_EMBEDDING_DIM`].
fn ensure_soccer_config_moment_tables(tx: &mut postgres::Transaction<'_>) -> Result<(), String> {
    tx.batch_execute(&format!(
        r#"
        create extension if not exists pgcrypto;
        create extension if not exists vector;
        create table if not exists des_soccer_config_moments (
          id uuid primary key default gen_random_uuid(),
          run_id uuid,
          experiment_id uuid,
          team varchar(8) not null,
          tick bigint not null,
          role varchar(16) not null,
          action varchar(64) not null,
          reward_micros bigint not null,
          nstep_return_micros bigint not null,
          value_micros bigint,
          embedding vector({dim}) not null,
          features real[] not null,
          created_at timestamptz not null default now(),
          deleted_at timestamptz,
          constraint des_soccer_config_moments_team_chk check (team in ('home','away')),
          constraint des_soccer_config_moments_features_len_chk
            check (array_length(features, 1) in ({legacy_features}, {features}))
        );
        alter table des_soccer_config_moments
          add column if not exists deleted_at timestamptz;
        -- Second (assigned-position / slot-aligned) corpus. Nullable so rows
        -- captured before this column existed remain valid; new inserts always
        -- populate both corpora.
        alter table des_soccer_config_moments
          add column if not exists embedding_assigned vector({dim});
        alter table des_soccer_config_moments
          add column if not exists features_assigned real[];
        alter table des_soccer_config_moments
          drop constraint if exists des_soccer_config_moments_features_len_chk;
        alter table des_soccer_config_moments
          add constraint des_soccer_config_moments_features_len_chk
            check (array_length(features, 1) in ({legacy_features}, {features}));
        alter table des_soccer_config_moments
          drop constraint if exists des_soccer_config_moments_features_assigned_len_chk;
        alter table des_soccer_config_moments
          add constraint des_soccer_config_moments_features_assigned_len_chk
            check (features_assigned is null
              or array_length(features_assigned, 1) in ({legacy_features}, {features}));
        create index if not exists des_soccer_config_moments_hnsw
          on des_soccer_config_moments using hnsw (embedding vector_cosine_ops);
        create index if not exists des_soccer_config_moments_hnsw_assigned
          on des_soccer_config_moments using hnsw (embedding_assigned vector_cosine_ops);
        create index if not exists des_soccer_config_moments_run_idx
          on des_soccer_config_moments (run_id);
        create index if not exists des_soccer_config_moments_live_created_idx
          on des_soccer_config_moments (experiment_id, created_at)
          where deleted_at is null;
        create index if not exists des_soccer_config_moments_deleted_idx
          on des_soccer_config_moments (deleted_at)
          where deleted_at is not null;
        "#,
        dim = SOCCER_MOMENT_EMBEDDING_DIM,
        legacy_features = CONFIG_FEATURE_DIM_V1,
        features = CONFIG_FEATURE_DIM
    ))
    .map_err(|err| format!("ensure soccer config moment tables: {err}"))
}

fn ensure_soccer_moment_embedding_tables(tx: &mut postgres::Transaction<'_>) -> Result<(), String> {
    tx.batch_execute(&format!(
        r#"
        create extension if not exists pgcrypto;
        create extension if not exists vector;
        create table if not exists des_soccer_moment_embeddings (
          id uuid primary key default gen_random_uuid(),
          run_id uuid,
          experiment_id uuid,
          team varchar(8) not null,
          tick bigint not null,
          action varchar(64) not null,
          reward_micros bigint not null,
          value_micros bigint,
          embedding vector({dim}) not null,
          created_at timestamptz not null default now(),
          deleted_at timestamptz,
          constraint des_soccer_moment_embeddings_team_chk check (team in ('home','away'))
        );
        alter table des_soccer_moment_embeddings
          add column if not exists deleted_at timestamptz;
        create index if not exists des_soccer_moment_embeddings_hnsw
          on des_soccer_moment_embeddings using hnsw (embedding vector_cosine_ops);
        create index if not exists des_soccer_moment_embeddings_run_idx
          on des_soccer_moment_embeddings (run_id);
        create index if not exists des_soccer_moment_embeddings_live_created_idx
          on des_soccer_moment_embeddings (experiment_id, created_at)
          where deleted_at is null;
        create index if not exists des_soccer_moment_embeddings_deleted_idx
          on des_soccer_moment_embeddings (deleted_at)
          where deleted_at is not null;
        "#,
        dim = SOCCER_MOMENT_EMBEDDING_DIM
    ))
    .map_err(|err| format!("ensure soccer moment embedding tables: {err}"))
}

fn ensure_soccer_learning_pass_metrics_table(
    tx: &mut postgres::Transaction<'_>,
) -> Result<(), String> {
    tx.batch_execute(
        r#"
        create table if not exists des_soccer_learning_pass_metrics (
          git_commit varchar(64) primary key,
          runs bigint not null default 0,
          passes_attempted bigint not null default 0,
          passes_completed bigint not null default 0,
          completed_pass_gain_yards_micros bigint not null default 0,
          pass_chains bigint not null default 0,
          pass_chain_gain_yards_micros bigint not null default 0,
          pass_chains_net_loss bigint not null default 0,
          shots_on_target bigint not null default 0,
          shots_after_pass bigint not null default 0,
          first_seen_at timestamptz default now() not null,
          updated_at timestamptz default now() not null,
          constraint des_soccer_learning_pass_metrics_counts_chk check (
            runs >= 0
            and passes_attempted >= 0
            and passes_completed >= 0
            and pass_chains >= 0
            and pass_chains_net_loss >= 0
            and shots_on_target >= 0
            and shots_after_pass >= 0
          )
        );

        create index if not exists des_soccer_learning_pass_metrics_updated_idx
          on des_soccer_learning_pass_metrics (updated_at desc);
        "#,
    )
    .map_err(|err| format!("ensure soccer learning pass metrics table: {err}"))
}

fn ensure_soccer_learning_set_play_tables(
    tx: &mut postgres::Transaction<'_>,
) -> Result<(), String> {
    tx.batch_execute(
        r#"
        create table if not exists des_soccer_learning_set_play_runs (
          run_id uuid primary key references des_soccer_learning_runs(id) on delete cascade,
          policy_version_id uuid not null references des_soccer_learning_policy_versions(id) on delete cascade,
          primary_restart varchar(40) not null,
          team varchar(8) not null,
          spot_x_micros bigint not null,
          spot_y_micros bigint not null,
          duration_seconds_micros bigint not null,
          episode_count integer not null,
          goals integer not null,
          goal_rate_micros bigint not null,
          first_window_goal_rate_micros bigint not null,
          last_window_goal_rate_micros bigint not null,
          goal_rate_delta_micros bigint not null,
          created_at timestamptz default now() not null,
          constraint des_soccer_learning_set_play_runs_restart_chk
            check (primary_restart in ('direct-free-kick', 'indirect-free-kick')),
          constraint des_soccer_learning_set_play_runs_team_chk
            check (team in ('home', 'away')),
          constraint des_soccer_learning_set_play_runs_duration_chk
            check (duration_seconds_micros >= 0),
          constraint des_soccer_learning_set_play_runs_episode_chk
            check (episode_count >= 0),
          constraint des_soccer_learning_set_play_runs_goals_chk
            check (goals >= 0),
          constraint des_soccer_learning_set_play_runs_goal_rate_chk
            check (goal_rate_micros between 0 and 1000000)
        );

        create table if not exists des_soccer_learning_set_play_restart_mix (
          run_id uuid not null references des_soccer_learning_set_play_runs(run_id) on delete cascade,
          ordinal integer not null,
          restart varchar(40) not null,
          primary key (run_id, ordinal),
          constraint des_soccer_learning_set_play_restart_mix_ordinal_chk
            check (ordinal >= 0),
          constraint des_soccer_learning_set_play_restart_mix_restart_chk
            check (restart in ('direct-free-kick', 'indirect-free-kick'))
        );

        create table if not exists des_soccer_learning_set_play_episode_metrics (
          run_id uuid not null references des_soccer_learning_set_play_runs(run_id) on delete cascade,
          episode_index integer not null,
          seed bigint not null,
          restart varchar(40) not null,
          routine varchar(80),
          scored boolean not null,
          score_delta_for_team integer not null,
          ticks bigint not null,
          simulated_seconds_micros bigint not null,
          policy_updates bigint not null,
          home_policy_entries integer not null,
          home_policy_target_entries integer not null,
          away_policy_entries integer not null,
          away_policy_target_entries integer not null,
          neural_training_steps integer not null,
          neural_samples bigint not null,
          neural_replay_samples integer not null,
          neural_last_loss_micros bigint,
          cumulative_goals integer not null,
          goal_rate_so_far_micros bigint not null,
          primary key (run_id, episode_index),
          constraint des_soccer_learning_set_play_episode_idx_chk
            check (episode_index >= 0),
          constraint des_soccer_learning_set_play_episode_seed_chk
            check (seed >= 0),
          constraint des_soccer_learning_set_play_episode_restart_chk
            check (restart in ('direct-free-kick', 'indirect-free-kick')),
          constraint des_soccer_learning_set_play_episode_ticks_chk
            check (ticks >= 0),
          constraint des_soccer_learning_set_play_episode_seconds_chk
            check (simulated_seconds_micros >= 0),
          constraint des_soccer_learning_set_play_episode_policy_updates_chk
            check (policy_updates >= 0),
          constraint des_soccer_learning_set_play_episode_entries_chk
            check (
              home_policy_entries >= 0
              and home_policy_target_entries >= 0
              and away_policy_entries >= 0
              and away_policy_target_entries >= 0
            ),
          constraint des_soccer_learning_set_play_episode_neural_chk
            check (
              neural_training_steps >= 0
              and neural_samples >= 0
              and neural_replay_samples >= 0
            ),
          constraint des_soccer_learning_set_play_episode_goals_chk
            check (cumulative_goals >= 0),
          constraint des_soccer_learning_set_play_episode_goal_rate_chk
            check (goal_rate_so_far_micros between 0 and 1000000)
        );

        create table if not exists des_soccer_learning_neural_run_metrics (
          run_id uuid primary key references des_soccer_learning_runs(id) on delete cascade,
          policy_version_id uuid not null references des_soccer_learning_policy_versions(id) on delete cascade,
          enabled boolean not null,
          backend varchar(32) not null,
          training_steps integer not null,
          samples bigint not null,
          pending_batches integer not null,
          dropped_batches integer not null,
          replay_samples integer not null,
          replay_capacity integer not null,
          parameter_count integer not null,
          target_clip_micros bigint not null,
          last_loss_micros bigint,
          average_loss_micros bigint,
          created_at timestamptz default now() not null,
          constraint des_soccer_learning_neural_run_backend_chk
            check (backend in ('inline', 'threaded')),
          constraint des_soccer_learning_neural_run_counts_chk
            check (
              training_steps >= 0
              and samples >= 0
              and pending_batches >= 0
              and dropped_batches >= 0
              and replay_samples >= 0
              and replay_capacity >= 0
              and parameter_count >= 0
            )
        );

        create index if not exists des_soccer_learning_set_play_episode_restart_idx
          on des_soccer_learning_set_play_episode_metrics (restart, scored, episode_index);

        create index if not exists des_soccer_learning_neural_run_steps_idx
          on des_soccer_learning_neural_run_metrics (training_steps desc, samples desc);
        "#,
    )
    .map_err(|err| format!("ensure soccer set-play learning tables: {err}"))?;
    Ok(())
}

fn insert_normalized_set_play_training_records(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    policy_version_id: &str,
    artifact: &SoccerSetPlayTrainingArtifact,
) -> Result<(), String> {
    let team = soccer_team_label(artifact.team);
    let primary_restart = artifact.restart.as_label();
    let spot_x_micros = soccer_learning_to_micros(artifact.spot.x);
    let spot_y_micros = soccer_learning_to_micros(artifact.spot.y);
    let duration_seconds_micros = soccer_learning_to_micros(artifact.duration_seconds);
    let episode_count = checked_i32(artifact.episodes.len());
    let goals = checked_i32(artifact.goals);
    let goal_rate_micros = soccer_learning_to_micros(artifact.goal_rate);
    let first_window_goal_rate_micros = soccer_learning_to_micros(artifact.first_window_goal_rate);
    let last_window_goal_rate_micros = soccer_learning_to_micros(artifact.last_window_goal_rate);
    let goal_rate_delta_micros = soccer_learning_to_micros(artifact.goal_rate_delta);

    tx.execute(
        r#"
        insert into des_soccer_learning_set_play_runs
          (
            run_id,
            policy_version_id,
            primary_restart,
            team,
            spot_x_micros,
            spot_y_micros,
            duration_seconds_micros,
            episode_count,
            goals,
            goal_rate_micros,
            first_window_goal_rate_micros,
            last_window_goal_rate_micros,
            goal_rate_delta_micros
          )
        values
          ($1::text::uuid, $2::text::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        "#,
        &[
            &run_id,
            &policy_version_id,
            &primary_restart,
            &team,
            &spot_x_micros,
            &spot_y_micros,
            &duration_seconds_micros,
            &episode_count,
            &goals,
            &goal_rate_micros,
            &first_window_goal_rate_micros,
            &last_window_goal_rate_micros,
            &goal_rate_delta_micros,
        ],
    )
    .map_err(|err| format!("insert soccer set-play run metrics: {err}"))?;

    for (ordinal, restart) in artifact.restarts.iter().enumerate() {
        let ordinal = checked_i32(ordinal);
        let restart_label = restart.as_label();
        tx.execute(
            r#"
            insert into des_soccer_learning_set_play_restart_mix
              (run_id, ordinal, restart)
            values
              ($1::text::uuid, $2, $3)
            "#,
            &[&run_id, &ordinal, &restart_label],
        )
        .map_err(|err| format!("insert soccer set-play restart mix: {err}"))?;
    }

    for episode in &artifact.episodes {
        let episode_index = checked_i32(episode.episode);
        let seed = checked_i64(episode.seed);
        let restart = episode.restart.as_label();
        let routine = episode
            .routine
            .map(|routine| routine.as_label().to_string());
        let ticks = checked_i64(episode.ticks);
        let simulated_seconds_micros = soccer_learning_to_micros(episode.simulated_seconds);
        let policy_updates = checked_i64(episode.policy_updates);
        let home_policy_entries = checked_i32(episode.home_policy_entries);
        let home_policy_target_entries = checked_i32(episode.home_policy_target_entries);
        let away_policy_entries = checked_i32(episode.away_policy_entries);
        let away_policy_target_entries = checked_i32(episode.away_policy_target_entries);
        let neural_training_steps = checked_i32(episode.neural_training_steps);
        let neural_samples = checked_i64(episode.neural_samples as u64);
        let neural_replay_samples = checked_i32(episode.neural_replay_samples);
        let neural_last_loss_micros = episode.neural_last_loss.map(soccer_learning_to_micros);
        let cumulative_goals = checked_i32(episode.cumulative_goals);
        let goal_rate_so_far_micros = soccer_learning_to_micros(episode.goal_rate_so_far);
        tx.execute(
            r#"
            insert into des_soccer_learning_set_play_episode_metrics
              (
                run_id,
                episode_index,
                seed,
                restart,
                routine,
                scored,
                score_delta_for_team,
                ticks,
                simulated_seconds_micros,
                policy_updates,
                home_policy_entries,
                home_policy_target_entries,
                away_policy_entries,
                away_policy_target_entries,
                neural_training_steps,
                neural_samples,
                neural_replay_samples,
                neural_last_loss_micros,
                cumulative_goals,
                goal_rate_so_far_micros
              )
            values
              (
                $1::text::uuid,
                $2,
                $3,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16,
                $17,
                $18,
                $19,
                $20
              )
            "#,
            &[
                &run_id,
                &episode_index,
                &seed,
                &restart,
                &routine,
                &episode.scored,
                &episode.score_delta_for_team,
                &ticks,
                &simulated_seconds_micros,
                &policy_updates,
                &home_policy_entries,
                &home_policy_target_entries,
                &away_policy_entries,
                &away_policy_target_entries,
                &neural_training_steps,
                &neural_samples,
                &neural_replay_samples,
                &neural_last_loss_micros,
                &cumulative_goals,
                &goal_rate_so_far_micros,
            ],
        )
        .map_err(|err| format!("insert soccer set-play episode metrics: {err}"))?;
    }

    let enabled = artifact.learning.neural_learning_enabled;
    let backend = artifact.learning.neural_learning_backend.as_str();
    let training_steps = checked_i32(artifact.learning.neural_learning_training_steps);
    let samples = checked_i64(artifact.learning.neural_learning_samples as u64);
    let pending_batches = checked_i32(artifact.learning.neural_learning_pending_batches);
    let dropped_batches = checked_i32(artifact.learning.neural_learning_dropped_batches);
    let replay_samples = checked_i32(artifact.learning.neural_learning_replay_samples);
    let replay_capacity = checked_i32(artifact.learning.neural_learning_replay_capacity);
    let parameter_count = checked_i32(artifact.learning.neural_learning_parameter_count);
    let target_clip_micros =
        soccer_learning_to_micros(artifact.learning.neural_learning_target_clip);
    let last_loss_micros = artifact
        .learning
        .neural_learning_last_loss
        .map(soccer_learning_to_micros);
    let average_loss_micros = artifact
        .learning
        .neural_learning_average_loss
        .map(soccer_learning_to_micros);
    tx.execute(
        r#"
        insert into des_soccer_learning_neural_run_metrics
          (
            run_id,
            policy_version_id,
            enabled,
            backend,
            training_steps,
            samples,
            pending_batches,
            dropped_batches,
            replay_samples,
            replay_capacity,
            parameter_count,
            target_clip_micros,
            last_loss_micros,
            average_loss_micros
          )
        values
          ($1::text::uuid, $2::text::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        "#,
        &[
            &run_id,
            &policy_version_id,
            &enabled,
            &backend,
            &training_steps,
            &samples,
            &pending_batches,
            &dropped_batches,
            &replay_samples,
            &replay_capacity,
            &parameter_count,
            &target_clip_micros,
            &last_loss_micros,
            &average_loss_micros,
        ],
    )
    .map_err(|err| format!("insert soccer neural run metrics: {err}"))?;

    Ok(())
}

/// The git commit the running binary was built from, used to key the per-commit learning
/// progress metrics. Injected by the build/deploy (the k8s Job sets `SOCCER_GIT_COMMIT`, the
/// same source the planner UI's `__PLANNER_GIT_COMMIT__` uses); falls back to `GIT_SHA`/
/// `GIT_COMMIT`, then to "dev" for a local run with no commit injected. Truncated to 64 chars
/// to fit the column.
pub fn soccer_learning_git_commit() -> String {
    [
        "SOCCER_GIT_COMMIT",
        "GIT_SHA",
        "GIT_COMMIT",
        "SOURCE_COMMIT",
    ]
    .into_iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
    // Char-safe truncation to the varchar(64) column width (`String::truncate` would panic on
    // a non-ASCII byte boundary; env values are untrusted).
    .map(|value| value.chars().take(64).collect::<String>())
    .unwrap_or_else(|| "dev".to_string())
}

fn soccer_learning_database_url() -> Option<String> {
    [
        "SOCCER_DATABASE_URL",
        "AGENT_TASKS_RDS_DATABASE_URL",
        "RDS_DATABASE_URL",
        "DATABASE_URL",
        "PG_DATABASE_URL",
    ]
    .into_iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn soccer_learning_env_nonnegative_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<i32>().ok())
        .filter(|value| *value >= 0)
        .unwrap_or(default)
}

fn soccer_pg_resume_max_policy_entries() -> Option<usize> {
    std::env::var(SOCCER_PG_RESUME_MAX_POLICY_ENTRIES_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

fn soccer_neural_snapshot_lookback_generations() -> i32 {
    soccer_learning_env_nonnegative_i32("SOCCER_POLICY_PROMOTION_BASELINE_LOOKBACK_GENERATIONS", 16)
}

fn soccer_neural_snapshot_min_sample_games() -> i32 {
    soccer_learning_env_nonnegative_i32("SOCCER_POLICY_PROMOTION_MIN_SAMPLE_GAMES", 8)
}

fn soccer_neural_snapshot_latest_qualified_trained() -> bool {
    soccer_learning_pg_env_flag("SOCCER_LIVE_POLICY_NEURAL_LATEST_QUALIFIED")
}

/// Session `statement_timeout` for soccer-learning PG connections. Defaults to "30s" so a slow or
/// stuck query can't hang a long-lived queue/server loop, but is overridable via
/// `SOCCER_PG_STATEMENT_TIMEOUT` (a Postgres interval such as "180s") for read-heavy clients like
/// the live server, which must load a large policy from a temporarily bloated table where the scan
/// can exceed 30s. The value is sanitized to `<int>[ms|s|min|h]` (falling back to "30s") since it
/// is interpolated into SQL.
fn soccer_learning_pg_statement_timeout() -> String {
    const DEFAULT: &str = "30s";
    let Some(raw) = std::env::var("SOCCER_PG_STATEMENT_TIMEOUT")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    else {
        return DEFAULT.to_string();
    };
    let digits: String = raw.chars().take_while(|c| c.is_ascii_digit()).collect();
    let unit = &raw[digits.len()..];
    if !digits.is_empty() && matches!(unit, "" | "ms" | "s" | "min" | "h") {
        raw
    } else {
        DEFAULT.to_string()
    }
}

fn soccer_policy_inline_prune_enabled() -> bool {
    let raw = std::env::var(SOCCER_POLICY_INLINE_PRUNE_ENV).ok();
    soccer_policy_inline_prune_enabled_from_env_value(raw.as_deref())
}

fn soccer_policy_inline_prune_enabled_from_env_value(raw: Option<&str>) -> bool {
    let value = raw.unwrap_or("").trim();
    value == "1"
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
}

fn soccer_learning_pg_persist_run_deltas_enabled() -> bool {
    let raw = std::env::var(SOCCER_PG_PERSIST_RUN_DELTAS_ENV).ok();
    soccer_learning_pg_persist_run_deltas_enabled_from_env_value(raw.as_deref())
}

fn soccer_learning_pg_persist_run_deltas_enabled_from_env_value(raw: Option<&str>) -> bool {
    let value = raw.unwrap_or("").trim();
    !(value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
        || value.eq_ignore_ascii_case("off"))
}

fn soccer_policy_version_full_entries_enabled() -> bool {
    let raw = std::env::var(SOCCER_POLICY_VERSION_FULL_ENTRIES_ENV).ok();
    soccer_policy_version_full_entries_enabled_from_env_value(raw.as_deref())
}

fn soccer_policy_version_full_entries_enabled_from_env_value(raw: Option<&str>) -> bool {
    let value = raw.unwrap_or("").trim();
    !(value == "0"
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
        || value.eq_ignore_ascii_case("off"))
}

fn soccer_learning_pg_should_verify_certificates(database_url: &str) -> bool {
    soccer_learning_pg_should_verify_certificates_with_override(
        database_url,
        soccer_learning_pg_env_flag("SOCCER_PG_TLS_INSECURE"),
    )
}

// Secure by default. Verification is relaxed ONLY when the operator explicitly
// opts out: the SOCCER_PG_TLS_INSECURE escape hatch, or an sslmode that by
// definition does not authenticate the server (`disable`/`allow`/`prefer`).
// `require`, `verify-ca`, `verify-full`, an unknown mode, and the no-sslmode
// default all verify. (This intentionally treats `require` more strictly than
// libpq, which encrypts without verifying.)
fn soccer_learning_pg_should_verify_certificates_with_override(
    database_url: &str,
    insecure_override: bool,
) -> bool {
    if insecure_override {
        return false;
    }
    match soccer_learning_pg_sslmode(database_url) {
        Some(mode)
            if mode.eq_ignore_ascii_case("disable")
                || mode.eq_ignore_ascii_case("allow")
                || mode.eq_ignore_ascii_case("prefer") =>
        {
            false
        }
        _ => true,
    }
}

fn soccer_learning_pg_env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            let value = value.trim();
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ) && !value.is_empty()
        })
        .unwrap_or(false)
}

// Optional CA bundle (PEM, one or more certificates) used to authenticate the
// server when the system trust store is insufficient — e.g. the AWS RDS global
// root on a minimal runner image.
fn soccer_learning_pg_root_certificates() -> Result<Vec<Certificate>, String> {
    let Some(path) = ["SOCCER_PG_SSLROOTCERT", "PGSSLROOTCERT"]
        .into_iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
    else {
        return Ok(Vec::new());
    };
    let pem = std::fs::read(&path)
        .map_err(|err| format!("read soccer learning postgres CA bundle {path}: {err}"))?;
    let mut certificates = Vec::new();
    for block in split_pem_certificates(&pem) {
        let certificate = Certificate::from_pem(&block).map_err(|err| {
            format!("parse soccer learning postgres CA certificate from {path}: {err}")
        })?;
        certificates.push(certificate);
    }
    if certificates.is_empty() {
        return Err(format!(
            "soccer learning postgres CA bundle {path} contained no PEM certificates"
        ));
    }
    Ok(certificates)
}

fn split_pem_certificates(pem: &[u8]) -> Vec<Vec<u8>> {
    const END_MARKER: &[u8] = b"-----END CERTIFICATE-----";
    let mut blocks = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while let Some(found) = find_subslice(&pem[idx..], END_MARKER) {
        let end = idx + found + END_MARKER.len();
        blocks.push(pem[start..end].to_vec());
        // Skip a trailing newline so the next block starts cleanly.
        idx = end;
        start = end;
    }
    blocks
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn soccer_learning_pg_connect_max_attempts() -> u32 {
    std::env::var("SOCCER_PG_CONNECT_MAX_ATTEMPTS")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .map(|value| value.clamp(1, 12))
        .unwrap_or(SOCCER_PG_CONNECT_DEFAULT_MAX_ATTEMPTS)
}

fn soccer_learning_pg_retry_backoff(attempt: u32) -> std::time::Duration {
    // Exponential backoff with a cap; attempt is 1-based.
    let exponent = attempt.saturating_sub(1).min(6);
    let millis = SOCCER_PG_CONNECT_BASE_BACKOFF_MILLIS
        .saturating_mul(1u64 << exponent)
        .min(SOCCER_PG_CONNECT_MAX_BACKOFF_MILLIS);
    std::time::Duration::from_millis(millis)
}

// A connect/IO failure (server unreachable, RDS failover in progress, dropped
// socket) is transient and worth retrying. A server-reported SQLSTATE is only
// transient for a known set (too-many-connections, admin shutdown, cannot-
// connect-now); auth/permission errors are permanent and must fail fast.
fn soccer_learning_pg_error_is_transient(err: &postgres::Error) -> bool {
    match err.as_db_error() {
        None => true,
        Some(db_error) => matches!(
            db_error.code().code(),
            "53300"
                | "53400"
                | "57P01"
                | "57P02"
                | "57P03"
                | "08000"
                | "08003"
                | "08006"
                | "08001"
                | "08004"
        ),
    }
}

fn soccer_learning_pg_error_context(err: &postgres::Error) -> String {
    let Some(db_error) = err.as_db_error() else {
        return err.to_string();
    };
    let mut parts = vec![
        format!("sqlstate={}", db_error.code().code()),
        format!("message={}", db_error.message()),
    ];
    if let Some(detail) = db_error.detail() {
        parts.push(format!("detail={detail}"));
    }
    if let Some(hint) = db_error.hint() {
        parts.push(format!("hint={hint}"));
    }
    if let Some(schema) = db_error.schema() {
        parts.push(format!("schema={schema}"));
    }
    if let Some(table) = db_error.table() {
        parts.push(format!("table={table}"));
    }
    if let Some(column) = db_error.column() {
        parts.push(format!("column={column}"));
    }
    if let Some(constraint) = db_error.constraint() {
        parts.push(format!("constraint={constraint}"));
    }
    parts.join(" ")
}

fn soccer_learning_pg_sslmode(database_url: &str) -> Option<&str> {
    let query = database_url.split_once('?')?.1;
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        key.eq_ignore_ascii_case("sslmode").then_some(value)
    })
}

fn state_hash(state_json: &Value) -> String {
    let raw = serde_json::to_string(state_json).unwrap_or_default();
    soccer_learning_fnv1a_128_hex(raw.as_bytes())
}

// 128-bit FNV-1a, rendered as 32 lowercase hex chars. Replaces the prior 64-bit
// variant to cut the birthday-collision probability of the content-addressed
// state_hash from ~2^-32 (at a few billion states) to negligible. 32 hex still
// satisfies the schema CHECK `state_hash ~ '^[a-f0-9]{16,32}$'`, so no migration
// is required; legacy 16-hex rows remain valid and are never compared by SQL to
// freshly written rows (run_deltas are write-only; the merge runs in Rust over
// reconstructed state keys). All three call sites in the crate use this exact
// function so engine- and persistence-computed hashes stay identical.
fn soccer_learning_fnv1a_128_hex(bytes: &[u8]) -> String {
    const OFFSET_BASIS: u128 = 0x6c62272e07bb0142_62b821756295c58d;
    const PRIME: u128 = 0x0000000001000000_000000000000013b;
    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("{hash:032x}")
}

fn checked_i32(value: impl TryInto<i64>) -> i32 {
    let value = value.try_into().unwrap_or(i64::MAX);
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn checked_i64(value: impl TryInto<u64>) -> i64 {
    let value = value.try_into().unwrap_or(u64::MAX);
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_neural_snapshot() -> SoccerNeuralNetworkSnapshot {
        SoccerNeuralNetworkSnapshot {
            input_dim: 2,
            output_dim: 1,
            parameter_count: 3,
            l2_norm: 0.5,
            layers: vec![crate::des::general::soccer::SoccerNeuralLayerSnapshot {
                activation: "linear".to_string(),
                weights: vec![vec![0.25, -0.25]],
                biases: vec![0.125],
            }],
            training_steps: 0,
            average_loss: None,
            target_popart: None,
            policy_head: None,
            line_depth_head: None,
        }
    }

    #[test]
    fn pg_vector_text_serializes_pgvector_form_and_scrubs_non_finite() {
        assert_eq!(pg_vector_text(&[1.0, 2.5, -3.0]), "[1,2.5,-3]");
        assert_eq!(pg_vector_text(&[]), "[]");
        // NaN/Inf must not break the `::vector` cast — scrubbed to 0.
        assert_eq!(pg_vector_text(&[f64::NAN, 1.0, f64::INFINITY]), "[0,1,0]");
    }

    #[test]
    fn moment_embedding_values_clause_offsets_and_param_limit() {
        assert_eq!(moment_embedding_values_clause(0), "");
        assert_eq!(
            moment_embedding_values_clause(1),
            "($1::text::uuid,$2::text::uuid,$3,$4,$5,$6,$7,$8::text::vector)"
        );
        assert_eq!(
            moment_embedding_values_clause(2),
            "($1::text::uuid,$2::text::uuid,$3,$4,$5,$6,$7,$8::text::vector),\
             ($1::text::uuid,$2::text::uuid,$9,$10,$11,$12,$13,$14::text::vector)"
        );
        // A full chunk must stay under Postgres's 65535 bound-parameter limit.
        let params = 2 + MOMENT_EMBEDDING_INSERT_CHUNK_ROWS * 6;
        assert!(params < 65535, "chunk uses {params} params");
    }

    #[test]
    fn config_moment_values_clause_offsets_and_param_limit() {
        assert_eq!(config_moment_values_clause(0), "");
        assert_eq!(
            config_moment_values_clause(1),
            "($1::text::uuid,$2::text::uuid,$3::text,$4::bigint,$5::text,$6::text,\
             $7::bigint,$8::bigint,$9::bigint,$10::text::vector,$11::real[],$12::text::vector,$13::real[])"
        );
        assert_eq!(
            config_moment_values_clause(2),
            "($1::text::uuid,$2::text::uuid,$3::text,$4::bigint,$5::text,$6::text,\
             $7::bigint,$8::bigint,$9::bigint,$10::text::vector,$11::real[],$12::text::vector,$13::real[]),\
             ($1::text::uuid,$2::text::uuid,$14::text,$15::bigint,$16::text,$17::text,\
             $18::bigint,$19::bigint,$20::bigint,$21::text::vector,$22::real[],$23::text::vector,$24::real[])"
        );
        // A full chunk must stay under Postgres's 65535 bound-parameter limit.
        let params = 2 + CONFIG_MOMENT_INSERT_CHUNK_ROWS * 11;
        assert!(params < 65535, "chunk uses {params} params");
    }

    #[test]
    fn config_moment_feature_width_tracks_holder_channel_migration() {
        assert_eq!(CONFIG_FEATURE_DIM_V1, 142);
        assert_eq!(crate::des::general::soccer::CONFIG_FEATURE_DIM_V2, 164);
        assert_eq!(CONFIG_FEATURE_DIM, 210);
        assert!(CONFIG_FEATURE_DIM > crate::des::general::soccer::CONFIG_FEATURE_DIM_V2);
    }

    #[test]
    fn default_moment_vector_retention_soft_deletes_after_ten_days() {
        assert_eq!(SOCCER_MOMENT_VECTOR_SOFT_DELETE_AFTER_DAYS, 10);
        assert_eq!(soccer_moment_vector_retention_limit_days(10), Some(10));
        assert_eq!(soccer_moment_vector_retention_limit_days(0), None);
        assert_eq!(soccer_moment_vector_retention_limit_days(-1), None);
    }

    #[test]
    fn moment_vector_search_queries_ignore_soft_deleted_rows() {
        for sql in [
            SOCCER_MOMENT_EMBEDDING_SEARCH_SQL,
            SOCCER_CONFIG_MOMENT_SEARCH_SQL,
        ] {
            assert!(
                sql.contains("deleted_at is null"),
                "live vector search must exclude soft-deleted rows: {sql}"
            );
            assert!(
                sql.contains("order by embedding <=> $1::text::vector"),
                "ANN ordering must stay vector-distance based: {sql}"
            );
        }
    }

    #[test]
    fn moment_vector_retention_queries_soft_delete_old_live_rows() {
        for sql in [
            SOCCER_MOMENT_EMBEDDING_SOFT_DELETE_SQL,
            SOCCER_CONFIG_MOMENT_SOFT_DELETE_SQL,
        ] {
            let normalized = sql.to_ascii_lowercase();
            assert!(normalized.starts_with("update "));
            assert!(!normalized.contains("delete from"));
            assert!(normalized.contains("set deleted_at = now()"));
            assert!(normalized.contains("where deleted_at is null"));
            assert!(normalized.contains("created_at < now() - ($2::bigint * interval '1 day')"));
            assert!(normalized.contains("experiment_id = $1::text::uuid"));
        }
    }

    #[test]
    fn moment_action_side_classifies_attack_vs_defense() {
        assert_eq!(soccer_moment_action_side("pass-forward"), Some(true));
        assert_eq!(soccer_moment_action_side("shoot"), Some(true));
        assert_eq!(soccer_moment_action_side("protect-ball"), None);
        assert_eq!(soccer_moment_action_side("tackle"), Some(false));
        assert_eq!(soccer_moment_action_side("defend-shape"), Some(false));
    }

    #[test]
    fn moment_neighbors_aggregate_favourable_outcomes_by_side() {
        let neighbors = vec![
            // Very similar, attacking, strongly favourable → drives attack up.
            SoccerMomentNeighbor {
                team: "home".to_string(),
                action: "shoot".to_string(),
                reward: 0.0,
                value: Some(3.0),
                tick: 10,
                distance: 0.05,
            },
            // Similar, defending, favourable → drives defense up.
            SoccerMomentNeighbor {
                team: "home".to_string(),
                action: "tackle".to_string(),
                reward: 0.0,
                value: Some(2.0),
                tick: 11,
                distance: 0.10,
            },
            // Orthogonal (distance ~1) → contributes ~nothing.
            SoccerMomentNeighbor {
                team: "home".to_string(),
                action: "pass".to_string(),
                reward: 0.0,
                value: Some(3.0),
                tick: 12,
                distance: 1.0,
            },
        ];
        let (attack, defense) = soccer_moment_neighbors_attack_defense(&neighbors);
        assert!((0.0..=1.0).contains(&attack) && (0.0..=1.0).contains(&defense));
        assert!(
            attack > 0.5,
            "favourable attacking neighbour should lift attack: {attack}"
        );
        assert!(
            defense > 0.5,
            "favourable defending neighbour should lift defense: {defense}"
        );

        // No neighbours / unfavourable outcomes ⇒ zero signal.
        assert_eq!(soccer_moment_neighbors_attack_defense(&[]), (0.0, 0.0));
        let unfavourable = vec![SoccerMomentNeighbor {
            team: "home".to_string(),
            action: "shoot".to_string(),
            reward: 0.0,
            value: Some(-5.0),
            tick: 1,
            distance: 0.0,
        }];
        assert_eq!(
            soccer_moment_neighbors_attack_defense(&unfavourable),
            (0.0, 0.0)
        );
    }

    #[test]
    fn soccer_learning_pg_sslmode_parses_query_param() {
        assert_eq!(
            soccer_learning_pg_sslmode("postgres://u:p@host/db?sslmode=require"),
            Some("require")
        );
        assert_eq!(
            soccer_learning_pg_sslmode(
                "postgres://u:p@host/db?connect_timeout=5&sslmode=verify-full"
            ),
            Some("verify-full")
        );
        assert_eq!(soccer_learning_pg_sslmode("postgres://u:p@host/db"), None);
    }

    #[test]
    fn soccer_learning_pg_verifies_certificates_by_default() {
        // Secure by default: no sslmode and the common `require` both verify.
        assert!(soccer_learning_pg_should_verify_certificates_with_override(
            "postgres://u:p@host/db",
            false
        ));
        assert!(soccer_learning_pg_should_verify_certificates_with_override(
            "postgres://u:p@host/db?sslmode=require",
            false
        ));
        assert!(soccer_learning_pg_should_verify_certificates_with_override(
            "postgres://u:p@host/db?sslmode=verify-ca",
            false
        ));
        assert!(soccer_learning_pg_should_verify_certificates_with_override(
            "postgres://u:p@host/db?sslmode=VERIFY-FULL",
            false
        ));
    }

    #[test]
    fn soccer_learning_pg_relaxes_only_on_explicit_opt_out() {
        // Non-authenticating sslmodes opt out.
        assert!(
            !soccer_learning_pg_should_verify_certificates_with_override(
                "postgres://u:p@host/db?sslmode=disable",
                false
            )
        );
        assert!(
            !soccer_learning_pg_should_verify_certificates_with_override(
                "postgres://u:p@host/db?sslmode=prefer",
                false
            )
        );
        assert!(
            !soccer_learning_pg_should_verify_certificates_with_override(
                "postgres://u:p@host/db?sslmode=allow",
                false
            )
        );
        // The insecure escape hatch overrides everything, including require.
        assert!(
            !soccer_learning_pg_should_verify_certificates_with_override(
                "postgres://u:p@host/db?sslmode=require",
                true
            )
        );
    }

    #[test]
    fn soccer_learning_pg_retry_backoff_is_bounded_and_monotonic() {
        let first = soccer_learning_pg_retry_backoff(1);
        let second = soccer_learning_pg_retry_backoff(2);
        assert!(second >= first);
        let capped = soccer_learning_pg_retry_backoff(100);
        assert!(capped.as_millis() as u64 <= SOCCER_PG_CONNECT_MAX_BACKOFF_MILLIS);
    }

    #[test]
    fn soccer_learning_pg_splits_pem_bundle_into_certificates() {
        let pem = b"-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n\
-----BEGIN CERTIFICATE-----\nBBBB\n-----END CERTIFICATE-----\n";
        let blocks = split_pem_certificates(pem);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].windows(4).any(|window| window == b"AAAA"));
        assert!(blocks[1].windows(4).any(|window| window == b"BBBB"));
    }

    #[test]
    fn policy_version_metrics_round_trip_neural_snapshot() {
        let snapshot = tiny_neural_snapshot();
        let mut tactical_learning = SoccerTacticalLearningWeights::default();
        tactical_learning.formation_lp_alignment_weight = 0.37;
        let metrics = soccer_policy_version_metrics(
            "evolution",
            1.25,
            Some(&snapshot),
            Some(&tactical_learning),
            None,
        )
        .expect("metrics");
        assert_eq!(metrics["fitness"], json!(1.25));
        assert_eq!(
            metrics["learningComponents"]["mdpPolicyEntries"],
            json!(true)
        );
        assert_eq!(
            metrics["learningComponents"]["pomdpStateKeyFeatures"],
            json!(true)
        );
        assert_eq!(
            metrics["learningComponents"]["mpcFormationLpWeights"],
            json!(true)
        );
        assert_eq!(
            metrics["learningComponents"]["lpAlignmentWeight"],
            json!(0.37)
        );
        assert_eq!(
            metrics["learningComponents"]["neuralNetworkSnapshot"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["sourceKind"],
            json!("evolution")
        );
        assert_eq!(
            metrics["learningProvenance"]["evolutionarySearch"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["geneticProgrammingTacticalSearch"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["tacticalLearningWeightsPersisted"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["neuralNetworkSnapshotPersisted"],
            json!(true)
        );

        let decoded =
            soccer_policy_version_neural_network_from_metrics(&metrics).expect("decode snapshot");
        let decoded = decoded.expect("snapshot present");
        assert_eq!(decoded.parameter_count, snapshot.parameter_count);
        assert_eq!(decoded.layers[0].weights, snapshot.layers[0].weights);
        assert_eq!(decoded.layers[0].biases, snapshot.layers[0].biases);

        let decoded_weights =
            soccer_policy_version_tactical_learning_from_values(&json!({}), &metrics)
                .expect("decode tactical learning")
                .expect("tactical learning present");
        assert_eq!(
            decoded_weights.attack_flank_lane_weight,
            tactical_learning.attack_flank_lane_weight
        );
        assert_eq!(
            decoded_weights.defense_contract_delta_weight,
            tactical_learning.defense_contract_delta_weight
        );
        assert_eq!(
            decoded_weights.formation_lp_alignment_weight,
            tactical_learning.formation_lp_alignment_weight
        );
    }

    #[test]
    fn policy_version_metrics_rejects_negative_neural_snapshot_norm() {
        let mut snapshot = tiny_neural_snapshot();
        snapshot.l2_norm = -0.1;
        let err = soccer_policy_version_metrics("evolution", 1.0, Some(&snapshot), None, None)
            .expect_err("negative neural snapshot norm should be rejected");
        assert!(err.contains("l2_norm"), "unexpected error: {err}");
    }

    #[test]
    fn policy_version_tactical_learning_falls_back_to_config_lp_weight() {
        let mut tactical_learning = SoccerTacticalLearningWeights::default();
        tactical_learning.formation_lp_alignment_weight = 0.42;
        let config = json!({
            "tacticalLearning": serde_json::to_value(&tactical_learning).expect("weights json")
        });
        let decoded = soccer_policy_version_tactical_learning_from_values(&config, &json!({}))
            .expect("decode tactical learning")
            .expect("tactical learning present");

        assert_eq!(decoded.formation_lp_alignment_weight, 0.42);
    }

    #[test]
    fn policy_version_metrics_mark_merge_without_evolutionary_search() {
        let metrics =
            soccer_policy_version_metrics("merge", 0.5, None, None, None).expect("metrics");

        assert_eq!(metrics["learningProvenance"]["sourceKind"], json!("merge"));
        assert_eq!(
            metrics["learningProvenance"]["evolutionarySearch"],
            json!(false)
        );
        assert_eq!(
            metrics["learningProvenance"]["geneticProgrammingTacticalSearch"],
            json!(false)
        );
        assert_eq!(
            metrics["learningProvenance"]["tacticalLearningWeightsPersisted"],
            json!(false)
        );
        assert_eq!(
            metrics["learningProvenance"]["neuralNetworkSnapshotPersisted"],
            json!(false)
        );
    }

    #[test]
    fn policy_version_metrics_stamp_policy_weight_fingerprint() {
        let mut metrics =
            soccer_policy_version_metrics("merge", 0.5, None, None, None).expect("metrics");
        let policies = SoccerTeamQPolicies::new(SoccerQPolicyOptions::default());

        stamp_soccer_policy_weight_metrics(&mut metrics, &policies);

        assert_eq!(
            metrics["learningComponents"]["policyFingerprint"],
            json!(soccer_team_q_policies_fingerprint(&policies))
        );
        assert_eq!(metrics["learningComponents"]["policyEntryCount"], json!(0));
        assert_eq!(
            metrics["learningProvenance"]["policyWeightsPersisted"],
            json!(true)
        );
        assert_eq!(
            soccer_policy_version_policy_fingerprint_from_metrics(&metrics),
            Some(soccer_team_q_policies_fingerprint(&policies))
        );
        assert_eq!(
            soccer_policy_version_policy_entry_count_from_metrics(&metrics),
            Some(0)
        );
    }

    #[test]
    fn policy_version_metrics_keep_search_parameters_json() {
        let metrics = soccer_policy_version_metrics(
            "evolution",
            0.75,
            None,
            Some(&SoccerTacticalLearningWeights::default()),
            Some(&json!({
                "algorithm": "evolutionary-genetic-programming",
                "completedGames": 100,
                "options": {
                    "mutationRate": 0.07,
                    "populationSize": 16
                }
            })),
        )
        .expect("metrics");

        assert_eq!(
            metrics["learningProvenance"]["searchParameters"]["algorithm"],
            json!("evolutionary-genetic-programming")
        );
        assert_eq!(
            metrics["learningProvenance"]["searchParameters"]["options"]["populationSize"],
            json!(16)
        );
        let decoded = soccer_policy_version_search_metadata_from_metrics(&metrics)
            .expect("decode search metadata")
            .expect("search metadata present");
        assert_eq!(
            decoded["algorithm"],
            json!("evolutionary-genetic-programming")
        );
        assert_eq!(decoded["options"]["mutationRate"], json!(0.07));
        assert_eq!(decoded["options"]["populationSize"], json!(16));
    }

    #[test]
    fn policy_version_metrics_detect_imported_evolutionary_search_metadata() {
        let metrics = soccer_policy_version_metrics(
            "server-import",
            0.75,
            None,
            Some(&SoccerTacticalLearningWeights::default()),
            Some(&json!({
                "policy": {
                    "algorithm": "evolutionary-policy-search"
                },
                "tactical": {
                    "algorithm": "evolutionary-genetic-programming-tactical-search"
                }
            })),
        )
        .expect("metrics");

        assert_eq!(
            metrics["learningProvenance"]["sourceKind"],
            json!("server-import")
        );
        assert_eq!(
            metrics["learningProvenance"]["evolutionarySearch"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["geneticProgrammingTacticalSearch"],
            json!(true)
        );
    }

    #[test]
    fn policy_version_metrics_detect_mutation_source_as_evolutionary_search() {
        let metrics = soccer_policy_version_metrics(
            "mutation",
            0.75,
            None,
            Some(&SoccerTacticalLearningWeights::default()),
            None,
        )
        .expect("metrics");

        assert_eq!(
            metrics["learningProvenance"]["sourceKind"],
            json!("mutation")
        );
        assert_eq!(
            metrics["learningProvenance"]["evolutionarySearch"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["geneticProgrammingTacticalSearch"],
            json!(true)
        );
    }

    #[test]
    fn policy_version_metrics_detect_genetic_programming_search_aliases() {
        let metrics = soccer_policy_version_metrics(
            "server-import",
            0.75,
            None,
            Some(&SoccerTacticalLearningWeights::default()),
            Some(&json!({
                "learningProvenance": {
                    "searchParameters": {
                        "searchAlgorithm": "geneticProgramming",
                        "searchFamily": "symbolicRegression"
                    }
                }
            })),
        )
        .expect("metrics");

        assert_eq!(
            metrics["learningProvenance"]["evolutionarySearch"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["geneticProgrammingTacticalSearch"],
            json!(true)
        );
        assert_eq!(
            metrics["learningProvenance"]["searchParameters"]["learningProvenance"]
                ["searchParameters"]["searchAlgorithm"],
            json!("geneticProgramming")
        );
    }

    #[test]
    fn policy_version_metrics_keep_plain_server_import_non_evolutionary() {
        let metrics = soccer_policy_version_metrics(
            "server-import",
            0.5,
            None,
            Some(&SoccerTacticalLearningWeights::default()),
            Some(&json!({
                "algorithm": "server-mdp-pomdp-neural-self-play",
                "source": "des-rs-http-training-endpoint"
            })),
        )
        .expect("metrics");

        assert_eq!(
            metrics["learningProvenance"]["evolutionarySearch"],
            json!(false)
        );
        assert_eq!(
            metrics["learningProvenance"]["geneticProgrammingTacticalSearch"],
            json!(false)
        );
        assert_eq!(
            metrics["learningProvenance"]["searchParameters"]["algorithm"],
            json!("server-mdp-pomdp-neural-self-play")
        );
    }

    #[test]
    fn postgres_rejects_non_finite_lp_alignment_weight() {
        let mut tactical_learning = SoccerTacticalLearningWeights::default();
        tactical_learning.formation_lp_alignment_weight = f64::INFINITY;

        assert!(
            validate_soccer_tactical_learning_weights_for_pg(&tactical_learning)
                .expect_err("non-finite LP alignment weight should be rejected")
                .contains("formationLpAlignmentWeight")
        );
    }

    #[test]
    fn postgres_rejects_malformed_neural_snapshot_metrics() {
        let metrics = json!({
            "neuralNetwork": {
                "inputDim": 2,
                "outputDim": 1,
                "parameterCount": 2,
                "l2Norm": 0.4,
                "layers": [{
                    "activation": "linear",
                    "weights": [[0.25]],
                    "biases": [0.1]
                }]
            }
        });

        assert!(soccer_policy_version_neural_network_from_metrics(&metrics)
            .expect_err("malformed neural snapshot should be rejected")
            .contains("expected 2"));
    }

    #[test]
    fn postgres_policy_state_json_preserves_mdp_pomdp_and_lp_bins() {
        let state_json = json!({
            "phase": "HomeBuildUp",
            "role": "Defender",
            "possessionRelative": -1,
            "ballZoneX": 3,
            "ballZoneY": 4,
            "scoreDiffBucket": 0,
            "teamBrainPressBin": 4,
            "teamBrainRiskBin": 2,
            "teamCentroidBallDistanceBin": 3,
            "teamSpreadBin": 2,
            "formationLpGuidance": true,
            "formationLpMoveBin": 3,
            "formationLpPairErrorBin": 2,
            "formationLpPressureBin": 4,
            "formationLpSpeedMatchBin": 1,
            "hasBall": false,
            "visibleBall": true,
            "shotLaneOpen": false,
            "visiblePassOptionsBin": 2,
            "ballDistanceBin": 1,
            "yardsToGoalBin": 4,
            "pressureBin": 3,
            "openSpaceBin": 2
        });
        let state: SoccerQStateKey =
            serde_json::from_value(state_json).expect("state key decodes with defaults");
        let persisted_state_json =
            serde_json::to_value(&state).expect("state key serializes for postgres jsonb");

        // Player pitch-grid cells and skill bins were removed from the Q-key; this
        // test now covers the retained team-brain / formation-LP bins.
        assert_eq!(persisted_state_json["teamBrainPressBin"], json!(4));
        assert_eq!(
            persisted_state_json["teamCentroidBallDistanceBin"],
            json!(3)
        );
        assert_eq!(persisted_state_json["formationLpGuidance"], json!(true));
        assert_eq!(persisted_state_json["formationLpMoveBin"], json!(3));
        assert_eq!(persisted_state_json["formationLpPairErrorBin"], json!(2));
        assert_eq!(persisted_state_json["formationLpPressureBin"], json!(4));
        assert_eq!(persisted_state_json["formationLpSpeedMatchBin"], json!(1));
        let hash = state_hash(&persisted_state_json);
        // 128-bit FNV-1a renders as 32 lowercase hex chars (still within the
        // schema CHECK `state_hash ~ '^[a-f0-9]{16,32}$'`).
        assert_eq!(hash.len(), 32);
        assert!(hash.chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn policy_version_metrics_record_branch_tip_retention() {
        let metrics = soccer_policy_version_metrics_with_retention(
            json!({ "fitness": 2.5 }),
            "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
            SOCCER_POLICY_RETENTION_BRANCH_TIP,
            true,
        );

        assert_eq!(
            metrics["postgresRetention"]["policyVersionId"],
            json!("aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa")
        );
        assert_eq!(
            metrics["postgresRetention"]["branchKey"],
            json!("bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb")
        );
        assert_eq!(
            metrics["postgresRetention"]["retentionKind"],
            json!("branch_tip")
        );
        assert_eq!(
            metrics["postgresRetention"]["fullEntriesRetained"],
            json!(true)
        );
    }

    #[test]
    fn policy_version_retention_skips_archived_and_rejected_entries() {
        assert!(soccer_policy_version_retains_full_entries_with_override(
            "active", false
        ));
        assert!(soccer_policy_version_retains_full_entries_with_override(
            "candidate",
            false
        ));
        assert!(!soccer_policy_version_retains_full_entries_with_override(
            "archived", false
        ));
        assert!(!soccer_policy_version_retains_full_entries_with_override(
            "rejected", false
        ));
        assert!(soccer_policy_version_retains_full_entries_with_override(
            "archived", true
        ));
        assert!(!soccer_policy_version_retains_full_entries_with_override(
            "rejected", true
        ));
    }

    #[test]
    fn policy_version_active_status_requires_promotion_sample_floor() {
        let one_game_metrics = soccer_policy_version_metrics(
            "merge",
            1.0,
            None,
            None,
            Some(&json!({
                "promotion": {
                    "gate": {
                        "sampleGames": 1,
                        "minSampleGames": 8
                    }
                }
            })),
        )
        .expect("metrics");
        assert_eq!(
            soccer_policy_version_status_after_promotion_sample_floor(
                SOCCER_POLICY_STATUS_ACTIVE,
                &one_game_metrics,
            ),
            SOCCER_POLICY_STATUS_ARCHIVED
        );

        let one_game_self_reported_floor_metrics = soccer_policy_version_metrics(
            "merge",
            1.0,
            None,
            None,
            Some(&json!({
                "promotion": {
                    "gate": {
                        "sampleGames": 1,
                        "minSampleGames": 1
                    }
                }
            })),
        )
        .expect("metrics");
        assert_eq!(
            soccer_policy_version_status_after_promotion_sample_floor(
                SOCCER_POLICY_STATUS_ACTIVE,
                &one_game_self_reported_floor_metrics,
            ),
            SOCCER_POLICY_STATUS_ARCHIVED
        );

        let eight_game_metrics = soccer_policy_version_metrics(
            "merge",
            1.0,
            None,
            None,
            Some(&json!({
                "promotion": {
                    "gate": {
                        "sampleGames": 8,
                        "minSampleGames": 8
                    }
                }
            })),
        )
        .expect("metrics");
        assert_eq!(
            soccer_policy_version_status_after_promotion_sample_floor(
                SOCCER_POLICY_STATUS_ACTIVE,
                &eight_game_metrics,
            ),
            SOCCER_POLICY_STATUS_ACTIVE
        );
    }

    #[test]
    fn inline_policy_prune_is_opt_in_for_learning_jobs() {
        assert!(!soccer_policy_inline_prune_enabled_from_env_value(None));
        assert!(!soccer_policy_inline_prune_enabled_from_env_value(Some("")));
        assert!(!soccer_policy_inline_prune_enabled_from_env_value(Some(
            "0"
        )));
        assert!(!soccer_policy_inline_prune_enabled_from_env_value(Some(
            "false"
        )));
        assert!(soccer_policy_inline_prune_enabled_from_env_value(Some("1")));
        assert!(soccer_policy_inline_prune_enabled_from_env_value(Some(
            "true"
        )));
        assert!(soccer_policy_inline_prune_enabled_from_env_value(Some(
            "YES"
        )));
        assert!(soccer_policy_inline_prune_enabled_from_env_value(Some(
            " on "
        )));
    }

    #[test]
    fn run_delta_persistence_defaults_on_and_can_be_disabled() {
        assert!(soccer_learning_pg_persist_run_deltas_enabled_from_env_value(None));
        assert!(soccer_learning_pg_persist_run_deltas_enabled_from_env_value(Some("")));
        assert!(soccer_learning_pg_persist_run_deltas_enabled_from_env_value(Some("1")));
        assert!(soccer_learning_pg_persist_run_deltas_enabled_from_env_value(Some("true")));
        assert!(!soccer_learning_pg_persist_run_deltas_enabled_from_env_value(Some("0")));
        assert!(!soccer_learning_pg_persist_run_deltas_enabled_from_env_value(Some("false")));
        assert!(!soccer_learning_pg_persist_run_deltas_enabled_from_env_value(Some(" off ")));
    }

    #[test]
    fn completed_run_retention_zero_disables_pruning() {
        assert_eq!(soccer_completed_run_retention_limit(0), None);
    }

    #[test]
    fn completed_run_retention_limit_is_bounded_for_postgres() {
        assert_eq!(soccer_completed_run_retention_limit(1), Some(1));
        assert_eq!(
            soccer_completed_run_retention_limit(usize::MAX),
            Some(i64::MAX)
        );
    }

    #[test]
    fn policy_entry_batch_placeholders_preserve_uuid_casts_and_offsets() {
        let mut completed_run_sql = String::new();
        append_completed_run_header_value_tuple(&mut completed_run_sql, 1);
        completed_run_sql.push_str(", ");
        append_completed_run_header_value_tuple(&mut completed_run_sql, 23);
        assert_eq!(
            completed_run_sql,
            "($1::text::uuid, $2::text::uuid, $3::text::uuid, $4::text::uuid, $5, $6, $7, 'completed', $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22), ($23::text::uuid, $24::text::uuid, $25::text::uuid, $26::text::uuid, $27, $28, $29, 'completed', $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44)"
        );

        let mut delta_sql = String::new();
        append_run_delta_value_tuple(&mut delta_sql, 1);
        delta_sql.push_str(", ");
        append_run_delta_value_tuple(&mut delta_sql, 18);
        assert_eq!(
            delta_sql,
            "($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17), ($18::text::uuid, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34)"
        );

        let mut action_sql = String::new();
        append_policy_entry_value_tuple(&mut action_sql, 1, false);
        action_sql.push_str(", ");
        append_policy_entry_value_tuple(&mut action_sql, 10, false);
        assert_eq!(
            action_sql,
            "($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9::text::uuid), ($10::text::uuid, $11, $12, $13, $14, $15, $16, $17, $18::text::uuid)"
        );

        let mut target_sql = String::new();
        append_policy_entry_value_tuple(&mut target_sql, 1, true);
        target_sql.push_str(", ");
        append_policy_entry_value_tuple(&mut target_sql, 15, true);
        assert_eq!(
            target_sql,
            "($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::text::uuid), ($15::text::uuid, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28::text::uuid)"
        );
    }

    #[test]
    fn soccer_learning_pg_batch_sizes_stay_under_postgres_parameter_limit() {
        assert!(
            SOCCER_COMPLETED_RUN_INSERT_BATCH_SIZE * SOCCER_COMPLETED_RUN_HEADER_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
        assert!(
            SOCCER_RUN_DELTA_INSERT_BATCH_SIZE * SOCCER_RUN_DELTA_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
        assert!(
            SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_ACTION_ENTRY_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
        assert!(
            SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE * SOCCER_POLICY_TARGET_ENTRY_PARAMETER_COUNT
                <= POSTGRES_MAX_QUERY_PARAMETERS
        );
        assert!(
            2 + PASS_OUTCOME_INSERT_CHUNK_ROWS * 3 <= POSTGRES_MAX_QUERY_PARAMETERS,
            "pass-outcome sidecar inserts share run_id/experiment_id and add 3 params per sample"
        );
    }

    #[test]
    fn policy_entry_insert_chunks_flush_at_bounded_capacity() {
        assert!(!policy_entry_insert_chunk_is_full(0));
        assert!(!policy_entry_insert_chunk_is_full(
            SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE - 1
        ));
        assert!(policy_entry_insert_chunk_is_full(
            SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE
        ));
        assert!(policy_entry_insert_chunk_is_full(
            SOCCER_POLICY_ENTRY_INSERT_BATCH_SIZE + 1
        ));
    }
}
