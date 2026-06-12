//! Release identity for the soccer engine and the engines it embeds.
//!
//! `build.rs` bakes this crate's git commit / commit date / build timestamp in
//! at compile time (`SOCCER_ENGINE_*`). The canonical [`BuildInfo`] type and its
//! JSON shape are reused from [`des_engine`] so every layer of the live soccer
//! stack reports the same fields.
//!
//! The live soccer UI fetches [`live_build_info_json`] from `GET /api/build` to
//! show which versions are actually running. When the engine is embedded in the
//! `dd-des-rs` web server, that server merges its own identity into the same
//! payload before returning it.

use des_engine::build_info::BuildInfo;
use serde::Serialize;
use serde_json::Value;

/// This crate's (`soccer_engine`) release identity.
pub fn build_info() -> BuildInfo {
    BuildInfo::new(
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("SOCCER_ENGINE_GIT_COMMIT"),
        env!("SOCCER_ENGINE_GIT_COMMIT_DATE"),
        env!("SOCCER_ENGINE_BUILD_TIMESTAMP"),
    )
}

/// Aggregate release identity for the live soccer stack as known from inside the
/// engine: the soccer engine itself plus the `des_engine` it links against. A
/// host (the web server) adds its own layer on top of this.
#[derive(Debug, Clone, Serialize)]
pub struct LiveBuildInfo {
    /// Schema tag so consumers can detect/version the payload.
    pub schema: &'static str,
    pub soccer_engine: BuildInfo,
    pub des_engine: BuildInfo,
}

/// The two-engine build identity reachable from within the engine.
pub fn live_build_info() -> LiveBuildInfo {
    LiveBuildInfo {
        schema: "dd.soccer.live.build.v1",
        soccer_engine: build_info(),
        des_engine: des_engine::build_info(),
    }
}

/// [`live_build_info`] as a JSON object, the wire form served at `/api/build`.
/// A host server can insert additional component keys (e.g. `web_server`).
pub fn live_build_info_json() -> Value {
    serde_json::to_value(live_build_info()).unwrap_or_else(|_| Value::Object(Default::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_reports_soccer_engine_identity() {
        let info = build_info();
        assert_eq!(info.name, "soccer_engine");
        assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
        assert!(!info.git_commit.is_empty());
        assert!(!info.commit_date.is_empty());
        assert!(!info.build_timestamp.is_empty());
    }

    #[test]
    fn live_build_info_json_carries_both_engine_layers() {
        let json = live_build_info_json();
        assert_eq!(json["schema"], "dd.soccer.live.build.v1");
        assert_eq!(json["soccer_engine"]["name"], "soccer_engine");
        assert_eq!(json["des_engine"]["name"], "des_engine");
        // Each layer exposes commit + timestamps.
        for layer in ["soccer_engine", "des_engine"] {
            assert!(json[layer]["git_commit"].is_string());
            assert!(json[layer]["git_commit_short"].is_string());
            assert!(json[layer]["commit_date"].is_string());
            assert!(json[layer]["build_timestamp"].is_string());
        }
    }
}
