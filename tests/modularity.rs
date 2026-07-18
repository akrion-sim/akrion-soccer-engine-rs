use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"))
}

fn collect_rust_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
    {
        let path = entry.expect("directory entry should be readable").path();
        if path.is_dir() {
            collect_rust_sources(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

#[test]
fn library_exposes_domain_and_telemetry_modules() {
    let library = read("src/lib.rs");
    assert!(library.contains("pub mod des;"));
    assert!(library.contains("pub use des::telemetry;"));

    let des = read("src/des/mod.rs");
    for module in [
        "general",
        "soccer_learning",
        "soccer_learning_pg",
        "soccer_planner",
        "streaming",
        "telemetry",
    ] {
        assert!(
            des.contains(&format!("pub mod {module};")),
            "src/des/mod.rs must expose the {module} module"
        );
    }
}

#[test]
fn lightweight_launchers_remain_thin() {
    for launcher in [
        "src/bin/main_soccer.rs",
        "src/bin/main_soccer_live.rs",
        "src/bin/main_soccer_live_5056.rs",
        "src/bin/main_soccer_planner.rs",
        "src/bin/main_soccer_rotation.rs",
        "src/bin/main_soccer_rotation_anim.rs",
        "src/bin/validate_soccer.rs",
    ] {
        let source = read(launcher);
        assert!(
            source.lines().count() <= 20,
            "{launcher} must remain a thin library launcher; found {} lines",
            source.lines().count()
        );
        assert!(
            source.contains("soccer_engine::"),
            "{launcher} must delegate to the soccer_engine library"
        );
    }
}

#[test]
fn deployed_learning_workloads_use_shared_telemetry() {
    for workload in [
        "src/bin/main_soccer_learning_server.rs",
        "src/bin/main_soccer_learning_queue.rs",
        "src/bin/main_soccer_learning_run.rs",
        "src/bin/main_soccer_tournament_run.rs",
    ] {
        let source = read(workload);
        assert!(
            source.contains("soccer_engine::telemetry::init_soccer_telemetry"),
            "{workload} must initialize the shared telemetry module"
        );
        assert!(
            source.contains("soccer_engine::telemetry::emit_process_start"),
            "{workload} must emit shared lifecycle metrics"
        );
        assert!(
            !source.contains("use opentelemetry")
                && !source.contains("opentelemetry_otlp")
                && !source.contains("tracing_subscriber"),
            "{workload} must not assemble its own telemetry pipeline"
        );
    }
}

#[test]
fn engine_sources_do_not_import_sqlx_directly() {
    let cargo_toml = read("Cargo.toml");
    assert!(
        !cargo_toml.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("sqlx =") || line.starts_with("sqlx=")
        }),
        "Cargo.toml must not declare SQLx directly"
    );

    let mut sources = Vec::new();
    collect_rust_sources(&repo_root().join("src"), &mut sources);
    for path in sources {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        assert!(
            !source.contains("use sqlx") && !source.contains("sqlx::"),
            "{} imports SQLx directly",
            path.display()
        );
    }
}
