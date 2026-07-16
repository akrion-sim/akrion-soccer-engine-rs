use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

const PRUNE_ENV: &str = "AKRION_CARGO_ARTIFACT_PRUNE";
const STALE_RCGU_AGE: Duration = Duration::from_secs(60 * 60);

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={PRUNE_ENV}");
    println!("cargo:rerun-if-env-changed=CARGO_INCREMENTAL");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    if env_flag_is_false(PRUNE_ENV) {
        return;
    }

    let Some(profile_dir) = cargo_profile_dir() else {
        println!(
            "cargo:warning=akrion artifact pruning skipped: cannot locate Cargo profile directory"
        );
        return;
    };

    let rustflags = format!(
        "{} {}",
        env::var("RUSTFLAGS").unwrap_or_default(),
        env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .replace('\u{1f}', " ")
    );
    let preserves_unpacked_outputs = rustflags.contains("split-debuginfo=unpacked")
        || rustflags.contains("save-temps=yes")
        || rustflags.contains("save-temps=y");

    let mut deleted_files = 0_u64;
    let mut reclaimed_bytes = 0_u64;

    // The repository profiles disable incremental compilation. Any existing
    // profile-local incremental tree (including dep-graph.bin/work-products.bin)
    // was produced under an older or explicitly overridden profile and cannot
    // be consumed by this build. Respect an explicit CARGO_INCREMENTAL=1 override.
    if !env_flag_is_true("CARGO_INCREMENTAL") {
        let incremental_dir = profile_dir.join("incremental");
        if incremental_dir.is_dir() {
            let bytes = directory_size(&incremental_dir);
            if bytes > 0 {
                match fs::remove_dir_all(&incremental_dir) {
                    Ok(()) => {
                        deleted_files = deleted_files.saturating_add(1);
                        reclaimed_bytes = reclaimed_bytes.saturating_add(bytes);
                    }
                    Err(err) => println!(
                        "cargo:warning=akrion artifact pruning could not remove {}: {err}",
                        incremental_dir.display()
                    ),
                }
            }
        }
    }

    // Cargo defaults to split-debuginfo=unpacked on macOS. Those *.rcgu.o files
    // are adjacent debug payloads after a completed build, but they also exist
    // briefly as live linker inputs. The age gate prevents a parallel rustc from
    // losing its inputs. Hashed rlib/rmeta outputs are retained because distinct
    // feature sets can still legitimately reference them.
    if !preserves_unpacked_outputs {
        let deps_dir = profile_dir.join("deps");
        if let Ok(entries) = fs::read_dir(&deps_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() || !has_suffix(path.file_name(), ".rcgu.o") {
                    continue;
                }
                let Ok(metadata) = entry.metadata() else {
                    continue;
                };
                let stale = metadata
                    .modified()
                    .ok()
                    .and_then(|modified| SystemTime::now().duration_since(modified).ok())
                    .is_some_and(|age| age >= STALE_RCGU_AGE);
                if !stale {
                    continue;
                }
                let bytes = metadata.len();
                match fs::remove_file(&path) {
                    Ok(()) => {
                        deleted_files = deleted_files.saturating_add(1);
                        reclaimed_bytes = reclaimed_bytes.saturating_add(bytes);
                    }
                    Err(err) => println!(
                        "cargo:warning=akrion artifact pruning could not remove {}: {err}",
                        path.display()
                    ),
                }
            }
        }
    }

    if deleted_files > 0 {
        println!(
            "cargo:warning=akrion artifact pruning removed {deleted_files} obsolete cache items ({}) before writing new outputs",
            human_bytes(reclaimed_bytes)
        );
    }
}

fn cargo_profile_dir() -> Option<PathBuf> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR")?);
    // OUT_DIR is <target>/<profile>/build/<package-hash>/out.
    out_dir.ancestors().nth(3).map(Path::to_path_buf)
}

fn env_flag_is_true(key: &str) -> bool {
    env::var(key).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_flag_is_false(key: &str) -> bool {
    env::var(key).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

fn has_suffix(name: Option<&OsStr>, suffix: &str) -> bool {
    name.and_then(OsStr::to_str)
        .is_some_and(|name| name.ends_with(suffix))
}

fn directory_size(root: &Path) -> u64 {
    let mut total = 0_u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file() {
                total = total
                    .saturating_add(entry.metadata().map(|metadata| metadata.len()).unwrap_or(0));
            }
        }
    }
    total
}

fn human_bytes(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else {
        format!("{bytes} bytes")
    }
}
