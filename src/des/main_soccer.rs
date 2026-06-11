//! Writes `out/soccer-sim.html` for the 2D live-match soccer prototype.

fn soccer_env_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn should_write_learning_artifacts(
    playback_only: Option<&str>,
    with_learning: Option<&str>,
) -> bool {
    if playback_only.is_some_and(soccer_env_truthy) {
        return false;
    }
    with_learning.is_some_and(soccer_env_truthy)
}

pub fn try_run() -> std::io::Result<()> {
    let playback_only = std::env::var("SOCCER_ARTIFACTS_PLAYBACK_ONLY").ok();
    let with_learning = std::env::var("SOCCER_ARTIFACTS_WITH_LEARNING").ok();
    if should_write_learning_artifacts(playback_only.as_deref(), with_learning.as_deref()) {
        let paths = crate::des::general::soccer::try_write_soccer_artifacts()?;
        crate::des::general::soccer::print_soccer_artifact_paths(&paths);
    } else {
        let paths = crate::des::general::soccer::try_write_soccer_playback_artifacts()?;
        crate::des::general::soccer::print_soccer_playback_artifact_paths(&paths);
    }
    Ok(())
}

pub fn run() {
    if let Err(err) = try_run() {
        eprintln!("main_soccer: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soccer_render_defaults_to_playback_only_artifacts() {
        assert!(!should_write_learning_artifacts(None, None));
        assert!(!should_write_learning_artifacts(Some("true"), Some("true")));
        assert!(should_write_learning_artifacts(None, Some("true")));
        assert!(should_write_learning_artifacts(Some("false"), Some("on")));
        assert!(!should_write_learning_artifacts(None, Some("false")));
    }
}
