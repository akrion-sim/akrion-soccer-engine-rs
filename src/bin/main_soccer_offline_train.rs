//! Sim-free offline "deeper learning" pass — the concrete answer to *can we learn another way?*
//!
//! Learning does NOT require running fresh self-play games. This binary refines the policy purely
//! from the accumulated Postgres corpus, running ZERO simulations:
//!   * the tabular team Q-policy is folded forward from the `des_soccer_learning_run_deltas`
//!     corpus (`merge_soccer_policy_deltas`, a prior-weighted merge of cross-game experience), and
//!   * the pass-completion head is trained on `des_soccer_pass_outcome_samples`.
//! The carried neural actor-critic snapshot is passed through UNCHANGED (offline actor-critic
//! training needs a per-transition store that does not exist yet — see the persistence-gap note),
//! so a persisted candidate stays live-loadable (never re-introduces the dropped-net bug).
//!
//! It is Hetzner-safe and DiskPressure-safe: no sim means no 12–15 GB/game footprint, so it can
//! "keep learning" even while the self-play learner is scaled to 0/0.
//!
//! SAFETY: DRY-RUN by default (reads + trains + reports, writes NOTHING). Set
//! `SOCCER_OFFLINE_TRAIN_COMMIT=1` to persist a candidate policy version — written as an
//! `archived` (non-active) candidate into a SEPARATE experiment (`SOCCER_OFFLINE_TRAIN_SLUG`,
//! default `soccer-offline-train`) so it can NEVER hijack the live self-play learner's
//! generation/active-pointer/promotion state. Point a live viewer's `SOCCER_EXPERIMENT_SLUG` at
//! that slug to watch the offline-refined policy.
//!
//! Run:
//!   # dry-run (safe): report what offline learning would do, no writes
//!   SOCCER_DATABASE_URL=... cargo run --release --features postgres-persistence \
//!     --bin main_soccer_offline_train
//!   # commit a candidate into the separate offline experiment
//!   SOCCER_DATABASE_URL=... SOCCER_OFFLINE_TRAIN_COMMIT=1 cargo run --release \
//!     --features postgres-persistence --bin main_soccer_offline_train

use std::io::{Error as IoError, ErrorKind};

use soccer_engine::des::general::soccer::{
    train_soccer_pass_completion_head, MatchConfig, SoccerQPolicyOptions,
};
use soccer_engine::des::soccer_learning::merge_soccer_policy_deltas;
use soccer_engine::des::soccer_learning_pg::SoccerLearningPgStore;
use uuid::Uuid;

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}
fn env_usize(key: &str, default: usize) -> usize {
    env_str(key).and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}
fn env_f64(key: &str, default: f64) -> f64 {
    env_str(key).and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}
fn env_flag(key: &str) -> bool {
    matches!(
        env_str(key).as_deref().map(|v| v.to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}
fn other(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::Other, message.into())
}

fn main() -> std::io::Result<()> {
    let source_slug = env_str("SOCCER_EXPERIMENT_SLUG")
        .unwrap_or_else(|| "soccer-self-play-k8s-overnight".to_string());
    let target_slug =
        env_str("SOCCER_OFFLINE_TRAIN_SLUG").unwrap_or_else(|| "soccer-offline-train".to_string());
    let commit = env_flag("SOCCER_OFFLINE_TRAIN_COMMIT");
    let delta_rows = env_usize("SOCCER_OFFLINE_TRAIN_DELTA_ROWS", 200_000);
    let prior_weight = env_f64("SOCCER_OFFLINE_TRAIN_PRIOR_WEIGHT", 1.0);
    let pass_limit = env_usize("SOCCER_OFFLINE_TRAIN_PASS_LIMIT", 100_000);
    let pass_epochs = env_usize("SOCCER_OFFLINE_TRAIN_PASS_EPOCHS", 12);
    let pass_lr = env_f64("SOCCER_OFFLINE_TRAIN_PASS_LR", 0.01);
    let seed = env_usize("SOCCER_OFFLINE_TRAIN_SEED", 2026) as u32;

    let mut store = match SoccerLearningPgStore::connect_from_env().map_err(other)? {
        Some(store) => store,
        None => {
            eprintln!("offline_train: no SOCCER_DATABASE_URL set — nothing to do");
            return Ok(());
        }
    };

    let config = MatchConfig::default();
    let source_experiment = store
        .ensure_experiment(&source_slug, &source_slug, &config)
        .map_err(other)?;

    // 1) Base policy to refine (carrying the neural net forward so a persisted candidate stays
    //    live-loadable). Full tabular core (min_visits = 0), best candidate included.
    let options = SoccerQPolicyOptions::default();
    let base = store
        .load_latest_policy_with_min_visits(
            &source_experiment,
            options.clone(),
            options.clone(),
            0,
            true,
        )
        .map_err(other)?
        .ok_or_else(|| other(format!("no policy version to refine in experiment '{source_slug}'")))?;
    let base_home_entries = base.policies.home.q_values.len();
    let base_away_entries = base.policies.away.q_values.len();
    println!(
        "offline_train: base gen={} home_entries={} away_entries={} neural_net={}",
        base.generation,
        base_home_entries,
        base_away_entries,
        base.neural_network.is_some()
    );

    // 2) Fold the accumulated cross-game tabular corpus into the policy — no games, just the
    //    logged experience re-weighted by `prior_weight` (>1 trusts the fresh corpus more).
    let delta = store
        .load_recent_completed_run_policy_delta(&source_experiment, delta_rows, None)
        .map_err(other)?;
    let delta_entries = delta.entries.len();
    let merged = if delta_entries > 0 {
        merge_soccer_policy_deltas(&base.policies, std::slice::from_ref(&delta), prior_weight)
            .map_err(other)?
    } else {
        base.policies.clone()
    };
    println!(
        "offline_train: tabular corpus delta_rows_loaded={} prior_weight={:.3} \
         merged_home_entries={} (Δ{:+}) merged_away_entries={} (Δ{:+})",
        delta_entries,
        prior_weight,
        merged.home.q_values.len(),
        merged.home.q_values.len() as i64 - base_home_entries as i64,
        merged.away.q_values.len(),
        merged.away.q_values.len() as i64 - base_away_entries as i64,
    );

    // 3) Train the pass-completion head on the accumulated pass-outcome corpus (offline SGD).
    let pass_samples = store
        .load_pass_outcome_samples(&source_experiment, pass_limit as i64)
        .map_err(other)?;
    match train_soccer_pass_completion_head(&pass_samples, seed, pass_epochs, pass_lr) {
        Some((_head, report)) => println!(
            "offline_train: pass-head trained samples={} steps={} epochs={} final_loss={:.5} accuracy={:.3}",
            report.samples, report.training_steps, report.epochs, report.final_loss, report.accuracy
        ),
        None => println!(
            "offline_train: pass-head training skipped (usable_samples=0 of {} loaded)",
            pass_samples.len()
        ),
    }

    // 4) Persist — or not. DRY-RUN prints the plan and stops; COMMIT writes an archived candidate
    //    into the SEPARATE offline experiment (never the live self-play one).
    if !commit {
        println!(
            "offline_train: DRY-RUN complete (no writes). Set SOCCER_OFFLINE_TRAIN_COMMIT=1 to \
             persist a candidate into experiment '{target_slug}'."
        );
        return Ok(());
    }

    let target_experiment = store
        .ensure_experiment(&target_slug, &target_slug, &config)
        .map_err(other)?;
    let version_id = Uuid::new_v4().to_string();
    let generation = base.generation.saturating_add(1);
    let label = format!(
        "offline-refined from {source_slug} gen{} (delta_rows={delta_entries}, prior={prior_weight:.2})",
        base.generation
    );
    let search_metadata = serde_json::json!({
        "offlineTraining": {
            "sourceExperiment": source_slug,
            "sourceGeneration": base.generation,
            "tabularDeltaRows": delta_entries,
            "priorWeight": prior_weight,
            "passSamples": pass_samples.len(),
            "noSimulation": true,
        }
    });
    store
        .insert_policy_version_with_id_and_neural_network_and_search_metadata(
            &version_id,
            &target_experiment,
            None,
            generation,
            &label,
            "merge",
            "archived",
            &config,
            options.clone(),
            options,
            &merged,
            0.0,
            base.neural_network.as_ref(),
            Some(&search_metadata),
        )
        .map_err(other)?;
    println!(
        "offline_train: COMMITTED candidate id={version_id} gen={generation} experiment='{target_slug}' \
         status=archived neural_net_carried={}",
        base.neural_network.is_some()
    );
    Ok(())
}
