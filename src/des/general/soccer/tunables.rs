//! Central registry of **tunable behavioral weights** for the soccer engine.
//!
//! Historically the engine was full of bare numeric literals — thresholds,
//! multipliers, margins — scattered across ~47k lines. Many are genuine
//! *tuning knobs* (how far is "close enough" to count a tackle, how much
//! movement counts as a dribble, ...) that we want to adjust without a
//! recompile and to override from the learning store.
//!
//! [`Tunables`] groups those knobs into one serde-serializable tree with three
//! override layers, applied lowest-to-highest:
//!
//! 1. **Compile-time defaults** — [`Tunables::default`], which reproduces the
//!    historical literal values exactly (so an unconfigured process is
//!    behaviorally identical to before this registry existed).
//! 2. **Environment** — a whole-tree `SOCCER_TUNABLES_JSON` blob *and/or*
//!    per-field `DD_SOCCER_TUNABLE__<dotted.path>=<json>` variables.
//! 3. **Postgres** — a JSON overlay supplied by the learning layer via
//!    [`init_tunables_with_overrides`] (e.g. the active policy's tuning row).
//!
//! Overlays are merged as JSON, so each layer only needs to specify the fields
//! it changes. Access the live values through [`tunables()`]; the first access
//! lazily materialises the env layers, so free functions deep in the engine can
//! read a knob with `tunables().tracking.moved_dt_multiplier` and no plumbing.
//!
//! Per-*match* knobs that must vary within a single process still belong in
//! [`crate::des::general::soccer::MatchConfig`]; this registry is the
//! process-wide home for the long tail of global literals.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Root of the tunable-weights tree. Grouped by subsystem; each group is a
/// plain serde struct with `#[serde(default)]` fields so a partial override
/// only names what it changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Tunables {
    /// Knobs for the frame-diff action *inference* used to label tracking data
    /// (`infer_tracking_action` and friends).
    pub tracking: TrackingInferenceTunables,
    /// Flank-cross legality + context-scoring weights
    /// (`flank_cross_context_score` / `_is_legal`).
    pub flank_cross: FlankCrossTunables,
    /// Discrete-event learning reward magnitudes (goal scored / conceded).
    pub reward: RewardTunables,
}

impl Default for Tunables {
    fn default() -> Self {
        Tunables {
            tracking: TrackingInferenceTunables::default(),
            flank_cross: FlankCrossTunables::default(),
            reward: RewardTunables::default(),
        }
    }
}

/// Thresholds the tracking-frame action inference uses to classify what a
/// player did between two frames. Defaults reproduce the previous inline
/// literals in `infer_tracking_action`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackingInferenceTunables {
    /// A player is judged to have actively *moved* (dribbled / repositioned)
    /// when their displacement exceeds `dt_seconds * this`. Previously the bare
    /// `1.2` in `moved > before.dt_seconds * 1.2`.
    pub moved_dt_multiplier: f64,
    /// Max yards-to-goal for a goalward ball move to be read as a `shoot`
    /// rather than a dribble. Previously `yards_to_goal <= 25.0`.
    pub shot_lane_max_yards_to_goal: f64,
    /// A defender who ends a frame this close to the ball after the opponent
    /// held it is credited with a `tackle`. Previously `< 3.8`.
    pub tackle_recover_max_distance_yards: f64,
    /// How many yards a defender must *close* on the ball within a frame to be
    /// credited with `defend`. Previously `after + 0.35 < before`.
    pub defend_closing_margin_yards: f64,
    /// Minimum open-space-score improvement for an on-possession off-ball player
    /// to be credited with finding `space`. Previously `+ 0.25`.
    pub space_improvement_threshold: f64,
}

impl Default for TrackingInferenceTunables {
    fn default() -> Self {
        TrackingInferenceTunables {
            moved_dt_multiplier: 1.2,
            shot_lane_max_yards_to_goal: 25.0,
            tackle_recover_max_distance_yards: 3.8,
            defend_closing_margin_yards: 0.35,
            space_improvement_threshold: 0.25,
        }
    }
}

/// Flank-cross legality threshold + the weights of the flank-cross context
/// score. Defaults reproduce the former `FLANK_CROSS_*` consts and the inline
/// literals in `flank_cross_context_score`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FlankCrossTunables {
    /// Minimum flank-lane score for a flank cross to be legal. Was
    /// `FLANK_CROSS_MIN_FLANK_SCORE`.
    pub min_flank_score: f64,
    /// Maximum yards-to-goal for a flank cross (legality gate + attacking-depth
    /// scale). Was `FLANK_CROSS_MAX_YARDS_TO_GOAL`.
    pub max_yards_to_goal: f64,
    /// Weight of perceived pressure in the pressure-release term.
    pub perceived_pressure_weight: f64,
    /// Weight of pressure urgency in the pressure-release term.
    pub pressure_urgency_weight: f64,
    /// Upper cap on the pressure-release term.
    pub pressure_release_cap: f64,
    /// Weight of the flank-lane component in the context score.
    pub flank_weight: f64,
    /// Weight of the attacking-depth component in the context score.
    pub attacking_depth_weight: f64,
}

impl Default for FlankCrossTunables {
    fn default() -> Self {
        FlankCrossTunables {
            min_flank_score: 0.52,
            max_yards_to_goal: 54.0,
            perceived_pressure_weight: 0.35,
            pressure_urgency_weight: 0.25,
            pressure_release_cap: 0.40,
            flank_weight: 0.62,
            attacking_depth_weight: 0.30,
        }
    }
}

/// Discrete-event reward magnitudes folded into the learning signal when a goal
/// is scored or conceded. Defaults reproduce the former inline literals in
/// `soccer_transition_reward_with_tactics`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardTunables {
    /// Reward added per goal the actor's team scores this transition. Was `100.0`.
    pub goal_scored_points: f64,
    /// Penalty when conceding, for a goalkeeper or defender (heavier — it's their
    /// job to prevent it). Was `8.0`.
    pub concede_keeper_defender_penalty: f64,
    /// Penalty when conceding, for an outfield non-defender. Was `2.0`.
    pub concede_outfield_penalty: f64,
}

impl Default for RewardTunables {
    fn default() -> Self {
        RewardTunables {
            goal_scored_points: 100.0,
            concede_keeper_defender_penalty: 8.0,
            concede_outfield_penalty: 2.0,
        }
    }
}

impl Tunables {
    /// Build the tunables from compile-time defaults plus the environment
    /// layers (`SOCCER_TUNABLES_JSON` then per-field `DD_SOCCER_TUNABLE__*`).
    /// Does not consult Postgres — see [`init_tunables_with_overrides`].
    pub fn from_env() -> Tunables {
        Tunables::from_overlays(env_overlays())
    }

    /// Build from defaults, then fold each JSON overlay on top in order
    /// (later overlays win). Each overlay only needs to name the fields it
    /// changes. Malformed overlays are skipped rather than panicking.
    pub fn from_overlays<I: IntoIterator<Item = Value>>(overlays: I) -> Tunables {
        let mut merged = serde_json::to_value(Tunables::default())
            .unwrap_or_else(|_| Value::Object(Default::default()));
        for overlay in overlays {
            merge_json(&mut merged, overlay);
        }
        serde_json::from_value(merged).unwrap_or_default()
    }
}

/// Recursively merge `overlay` into `base`: objects are merged key-by-key, any
/// other value (scalar, array) replaces wholesale. Mirrors how a config patch
/// should behave.
fn merge_json(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            for (key, value) in overlay_map {
                merge_json(base_map.entry(key).or_insert(Value::Null), value);
            }
        }
        (base_slot, overlay) => *base_slot = overlay,
    }
}

/// Collect the environment override layers, lowest-priority first:
/// 1. `SOCCER_TUNABLES_JSON` — a full or partial JSON object.
/// 2. each `DD_SOCCER_TUNABLE__<dotted.path>=<json-or-string>` as a one-field
///    overlay (so `DD_SOCCER_TUNABLE__tracking.moved_dt_multiplier=1.25` sets
///    exactly that leaf).
fn env_overlays() -> Vec<Value> {
    let mut overlays = Vec::new();

    if let Ok(blob) = std::env::var("SOCCER_TUNABLES_JSON") {
        match serde_json::from_str::<Value>(&blob) {
            Ok(value) => overlays.push(value),
            Err(err) => eprintln!("soccer tunables: ignoring malformed SOCCER_TUNABLES_JSON: {err}"),
        }
    }

    const FIELD_PREFIX: &str = "DD_SOCCER_TUNABLE__";
    let mut field_overrides: Vec<(String, String)> = std::env::vars()
        .filter_map(|(key, value)| key.strip_prefix(FIELD_PREFIX).map(|p| (p.to_string(), value)))
        .collect();
    // Deterministic application order regardless of env iteration order.
    field_overrides.sort();
    for (path, raw) in field_overrides {
        // Parse the value as JSON (numbers/bools/objects), falling back to a
        // bare string for unquoted text like `DD_SOCCER_TUNABLE__x=foo`.
        let leaf = serde_json::from_str::<Value>(&raw).unwrap_or(Value::String(raw));
        overlays.push(nest_dotted_path(&path, leaf));
    }

    overlays
}

/// Turn a dotted path + leaf value into a nested JSON object,
/// e.g. `("tracking.moved_dt_multiplier", 1.25)` =>
/// `{"tracking": {"moved_dt_multiplier": 1.25}}`.
fn nest_dotted_path(path: &str, leaf: Value) -> Value {
    let mut value = leaf;
    for segment in path.split('.').rev() {
        let mut map = serde_json::Map::new();
        map.insert(segment.to_string(), value);
        value = Value::Object(map);
    }
    value
}

static TUNABLES: OnceLock<Tunables> = OnceLock::new();

/// The live, process-wide tunables. Lazily materialised from the environment
/// layers on first access. If Postgres overrides are wanted they must be
/// installed via [`init_tunables_with_overrides`] *before* the first call here.
pub fn tunables() -> &'static Tunables {
    TUNABLES.get_or_init(Tunables::from_env)
}

/// Install the process tunables built from defaults + environment + the given
/// Postgres/extra JSON overlays (applied last, highest priority). Returns
/// `true` if this call performed the initialisation, `false` if the registry
/// was already initialised (in which case the overrides are ignored — call
/// this once at startup before any `tunables()` read).
pub fn init_tunables_with_overrides<I: IntoIterator<Item = Value>>(pg_overlays: I) -> bool {
    let mut overlays = env_overlays();
    overlays.extend(pg_overlays);
    let built = Tunables::from_overlays(overlays);
    let mut installed = false;
    let _ = TUNABLES.get_or_init(|| {
        installed = true;
        built
    });
    installed
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_historical_literals() {
        let t = Tunables::default();
        assert_eq!(t.tracking.moved_dt_multiplier, 1.2);
        assert_eq!(t.tracking.shot_lane_max_yards_to_goal, 25.0);
        assert_eq!(t.tracking.tackle_recover_max_distance_yards, 3.8);
        assert_eq!(t.tracking.defend_closing_margin_yards, 0.35);
        assert_eq!(t.tracking.space_improvement_threshold, 0.25);
    }

    #[test]
    fn partial_overlay_only_touches_named_fields() {
        let t = Tunables::from_overlays([json!({
            "tracking": { "moved_dt_multiplier": 1.5 }
        })]);
        assert_eq!(t.tracking.moved_dt_multiplier, 1.5);
        // Untouched fields keep their defaults.
        assert_eq!(t.tracking.tackle_recover_max_distance_yards, 3.8);
    }

    #[test]
    fn later_overlay_wins() {
        let t = Tunables::from_overlays([
            json!({ "tracking": { "moved_dt_multiplier": 1.5 } }),
            json!({ "tracking": { "moved_dt_multiplier": 2.0 } }),
        ]);
        assert_eq!(t.tracking.moved_dt_multiplier, 2.0);
    }

    #[test]
    fn dotted_path_nests_correctly() {
        let nested = nest_dotted_path("tracking.moved_dt_multiplier", json!(1.25));
        assert_eq!(nested, json!({ "tracking": { "moved_dt_multiplier": 1.25 } }));
    }

    #[test]
    fn merge_replaces_scalars_and_merges_objects() {
        let mut base = json!({ "a": { "x": 1, "y": 2 }, "b": 3 });
        merge_json(&mut base, json!({ "a": { "y": 9 }, "c": 4 }));
        assert_eq!(base, json!({ "a": { "x": 1, "y": 9 }, "b": 3, "c": 4 }));
    }
}
