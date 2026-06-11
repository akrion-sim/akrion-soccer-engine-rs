use std::env;
use std::error::Error;
use std::fmt;

use serde_json::{json, Value};

const DEFAULT_DEPLOYMENT: &str = "dd-soccer-learning-rds-smoke-20260604a";
const DEFAULT_NAMESPACE: &str = "default";
const DEFAULT_SOURCE_REF: &str = "main";
const DEFAULT_GAMES: &str = "25";
const DEFAULT_PARALLEL_GAMES: &str = "5";

#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderArgs {
    deployment: String,
    namespace: String,
    patched_at: String,
    run_id: String,
    source_build: String,
    source_sha256: String,
    source_auth_header: String,
    source_ref: String,
    source_commit: String,
    games: String,
    parallel_games: String,
}

#[derive(Debug, PartialEq, Eq)]
struct ArgError(String);

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ArgError {}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} --patched-at <ts> --run-id <id> --source-build <id> --source-sha256 <sha> [--deployment <name>] [--namespace <ns>] [--source-ref <ref>] [--source-commit <sha>] [--source-auth-header <header>] [--games <n>] [--parallel-games <n>]"
    )
}

fn parse_args_from<I, S>(program: &str, args: I) -> Result<RenderArgs, ArgError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut deployment = DEFAULT_DEPLOYMENT.to_string();
    let mut namespace = DEFAULT_NAMESPACE.to_string();
    let mut patched_at = None::<String>;
    let mut run_id = None::<String>;
    let mut source_build = None::<String>;
    let mut source_sha256 = None::<String>;
    let mut source_auth_header = String::new();
    let mut source_ref = DEFAULT_SOURCE_REF.to_string();
    let mut source_commit = String::new();
    let mut games = DEFAULT_GAMES.to_string();
    let mut parallel_games = DEFAULT_PARALLEL_GAMES.to_string();
    let mut values = args.into_iter().map(Into::into).peekable();

    while let Some(raw) = values.next() {
        if raw == "-h" || raw == "--help" {
            return Err(ArgError(usage(program)));
        }
        let (key, inline_value) = if let Some((key, value)) = raw.split_once('=') {
            (key.to_string(), Some(value.to_string()))
        } else {
            (raw, None)
        };
        let value = if let Some(value) = inline_value {
            value
        } else {
            let value = values
                .next()
                .ok_or_else(|| ArgError(format!("{key} requires a value\n{}", usage(program))))?;
            if value.starts_with("--") {
                return Err(ArgError(format!(
                    "{key} requires a value\n{}",
                    usage(program)
                )));
            }
            value
        };
        match key.as_str() {
            "--deployment" => deployment = value,
            "--namespace" => namespace = value,
            "--patched-at" => patched_at = Some(value),
            "--run-id" => run_id = Some(value),
            "--source-build" => source_build = Some(value),
            "--source-sha256" => source_sha256 = Some(value),
            "--source-auth-header" => source_auth_header = value,
            "--source-ref" => source_ref = value,
            "--source-commit" => source_commit = value,
            "--games" => games = value,
            "--parallel-games" => parallel_games = value,
            _ => {
                return Err(ArgError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )));
            }
        }
    }

    let source_sha256 = source_sha256
        .ok_or_else(|| ArgError(format!("--source-sha256 is required\n{}", usage(program))))?;
    if source_commit.is_empty() {
        if let Some(commit) = source_sha256.strip_prefix("git:") {
            source_commit = commit.to_string();
        }
    }

    Ok(RenderArgs {
        deployment,
        namespace,
        patched_at: patched_at
            .ok_or_else(|| ArgError(format!("--patched-at is required\n{}", usage(program))))?,
        run_id: run_id
            .ok_or_else(|| ArgError(format!("--run-id is required\n{}", usage(program))))?,
        source_build: source_build
            .ok_or_else(|| ArgError(format!("--source-build is required\n{}", usage(program))))?,
        source_sha256,
        source_auth_header,
        source_ref,
        source_commit,
        games,
        parallel_games,
    })
}

fn env_value(name: &str, value: impl Into<String>) -> Value {
    json!({
        "name": name,
        "value": value.into(),
    })
}

fn env_value_from(name: &str, value_from: Value) -> Value {
    json!({
        "name": name,
        "valueFrom": value_from,
    })
}

fn bootstrap_script(args: &RenderArgs) -> String {
    format!(
        r#"set -euo pipefail
export PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git curl ca-certificates
fi
run_tag={run_id}
source_build={source_build}
source_sha={source_sha256}
source_ref={source_ref}
source_commit={source_commit}
work="/tmp/${{run_tag}}-src-$(date -u +%Y%m%dT%H%M%SZ)-$$"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "runner=$(hostname)"
echo "source_build=${{source_build}}"
echo "source_ref=${{source_ref}}"
echo "source_commit_expected=${{source_commit:-none}}"
echo "source_work=${{work}}"
if [ -n "${{source_commit}}" ]; then
  git init "${{work}}"
  cd "${{work}}"
  git remote add origin https://github.com/ORESoftware/discrete-event-system.rs.git
  git fetch --depth 1 origin "${{source_commit}}" || git fetch --depth 50 origin "${{source_ref}}"
  git checkout --detach "${{source_commit}}"
else
  git clone --depth 1 --branch "${{source_ref}}" https://github.com/ORESoftware/discrete-event-system.rs.git "${{work}}"
  cd "${{work}}"
fi
actual="$(git rev-parse HEAD)"
echo "source_commit_actual=${{actual}}"
if [ -n "${{source_commit}}" ]; then
  test "${{actual}}" = "${{source_commit}}"
fi
exec bash k8s/run-soccer-learning-rds-smoke.sh
"#,
        run_id = args.run_id,
        source_build = args.source_build,
        source_sha256 = args.source_sha256,
        source_ref = args.source_ref,
        source_commit = args.source_commit
    )
}

fn render_patch(args: &RenderArgs) -> Value {
    let rds_secret_ref = json!({
        "secretKeyRef": {
            "name": "dd-remote-rest-api-secrets",
            "key": "RDS_DATABASE_URL",
        }
    });

    json!({
        "metadata": {
            "annotations": {
                "codex.ores/patched-at": args.patched_at,
                "codex.ores/run-status": "patched",
                "codex.ores/run-stage": "waiting-for-rollout",
                "codex.ores/run-id": args.run_id,
                "codex.ores/source-build": args.source_build,
                "codex.ores/source-commit": args.source_commit,
                "codex.ores/source-sha256": args.source_sha256,
            }
        },
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "codex.ores/restarted-at": args.patched_at,
                        "codex.ores/run-id": args.run_id,
                        "codex.ores/source-build": args.source_build,
                        "codex.ores/source-commit": args.source_commit,
                    }
                },
                "spec": {
                    "serviceAccountName": "dd-remote-rest-api",
                    "automountServiceAccountToken": true,
                    "containers": [
                        {
                            "name": "soccer-learning",
                            "image": "docker.io/library/rust:1.90-bookworm",
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc"],
                            "args": [bootstrap_script(args)],
                            "env": [
                                env_value("HOME", "/tmp"),
                                env_value("CARGO_HOME", "/tmp/cargo"),
                                env_value("CARGO_TARGET_DIR", format!("/tmp/{}-target", args.run_id)),
                                env_value("CARGO_BUILD_JOBS", "2"),
                                env_value_from("SOCCER_DATABASE_URL", rds_secret_ref),
                                env_value("SOCCER_RUN_ID", &args.run_id),
                                env_value("SOCCER_EXPERIMENT_SLUG", "soccer-self-play-k8s-smoke"),
                                env_value("SOCCER_EXPERIMENT_NAME", "Soccer self-play k8s smoke"),
                                env_value("SOCCER_GAMES", &args.games),
                                env_value("SOCCER_PARALLEL_GAMES", &args.parallel_games),
                                env_value("SOCCER_HALVES", "2"),
                                env_value("SOCCER_HALF_MINUTES", "45"),
                                env_value("SOCCER_MINUTES", "90"),
                                env_value("SOCCER_PERIOD_BREAK_RECOVERY_SECONDS", "900"),
                                env_value("SOCCER_DT_SECONDS", "5"),
                                env_value("SOCCER_LEARNING_INTERVAL_TICKS", "4"),
                                env_value("SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES", "5"),
                                env_value("SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES", "5"),
                                env_value("SOCCER_POSTGRES_COMPLETED_RUN_RETENTION_GAMES", "200"),
                                env_value("SOCCER_POSTGRES_ASYNC_BATCH_QUEUE", "4"),
                                env_value("SOCCER_EVOLUTION_ENABLED", "true"),
                                env_value("SOCCER_EVOLUTION_INTERVAL_GAMES", "5"),
                                env_value("SOCCER_EVOLUTION_ELITE_GAMES", "4"),
                                env_value("SOCCER_CHECKPOINT_INTERVAL_GAMES", "0"),
                                env_value("SOCCER_GAME_ARTIFACT_MODE", "summary"),
                                env_value("SOCCER_WRITE_GAME_ARTIFACTS", "false"),
                                env_value("SOCCER_WRITE_FINAL_ARTIFACTS", "false"),
                                env_value("SOCCER_WRITE_FINAL_POLICY_ARTIFACT", "false"),
                                env_value("SOCCER_WRITE_CHECKPOINT_ARTIFACTS", "false"),
                                env_value("SOCCER_WRITE_EPISODE_LOG", "false"),
                                env_value("SOCCER_REQUIRE_POSTGRES", "true"),
                                env_value("SOCCER_RUN_DIR", format!("/tmp/{}", args.run_id)),
                                env_value("SOCCER_ARTIFACT_PATH", "/dev/null"),
                                env_value("SOCCER_CHECKPOINT_ARTIFACT_PATH", "/dev/null"),
                                env_value("SOCCER_EPISODE_LOG_PATH", "/dev/null"),
                                env_value("SOCCER_LEARNED_PARAMS_PATH", "/dev/null"),
                                env_value("SOCCER_K8S_DEPLOYMENT_NAME", &args.deployment),
                                env_value("SOCCER_K8S_NAMESPACE", &args.namespace),
                                env_value("SOCCER_SOURCE_BUILD", &args.source_build),
                                env_value("SOCCER_SOURCE_SHA256", &args.source_sha256),
                                env_value("SOCCER_SOURCE_COMMIT", &args.source_commit),
                                env_value("SOCCER_SOURCE_AUTH_HEADER", &args.source_auth_header),
                                env_value("SOCCER_KEEPALIVE_SECONDS", "86400"),
                            ],
                            "resources": {
                                "requests": {"cpu": "1500m", "memory": "2Gi"},
                                "limits": {"cpu": "5", "memory": "8Gi"},
                            },
                            "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
                        }
                    ],
                    "volumes": [{"name": "tmp", "emptyDir": {}}],
                },
            }
        },
    })
}

fn main() {
    let mut raw_args = env::args();
    let program = raw_args
        .next()
        .unwrap_or_else(|| "render_soccer_learning_rds_smoke_patch".to_string());
    let args = raw_args.collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage(&program));
        return;
    }
    match parse_args_from(&program, args) {
        Ok(args) => {
            let patch = render_patch(&args);
            println!(
                "{}",
                serde_json::to_string(&patch).expect("serialize patch")
            );
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_args() -> Vec<&'static str> {
        vec![
            "--patched-at",
            "2026-06-05T12:00:00Z",
            "--run-id",
            "run-a",
            "--source-build",
            "build-a",
            "--source-sha256",
            "git:abc123",
        ]
    }

    #[test]
    fn derives_source_commit_from_git_sha_marker() {
        let args = parse_args_from("renderer", required_args()).expect("args");
        assert_eq!(args.source_commit, "abc123");
    }

    #[test]
    fn explicit_source_commit_wins_over_git_sha_marker() {
        let mut raw = required_args();
        raw.extend(["--source-commit", "def456"]);
        let args = parse_args_from("renderer", raw).expect("args");
        assert_eq!(args.source_commit, "def456");
    }

    #[test]
    fn render_patch_sets_postgres_and_evolution_env() {
        let args = parse_args_from(
            "renderer",
            [
                "--patched-at=2026-06-05T12:00:00Z",
                "--run-id=run-a",
                "--source-build=build-a",
                "--source-sha256=git:abc123",
                "--games=12",
                "--parallel-games=3",
            ],
        )
        .expect("args");
        let patch = render_patch(&args);
        let env = patch["spec"]["template"]["spec"]["containers"][0]["env"]
            .as_array()
            .expect("env array");
        let value_for = |name: &str| {
            env.iter()
                .find(|entry| entry["name"] == name)
                .and_then(|entry| entry["value"].as_str())
                .unwrap_or("")
        };

        assert_eq!(value_for("SOCCER_GAMES"), "12");
        assert_eq!(value_for("SOCCER_PARALLEL_GAMES"), "3");
        assert_eq!(value_for("SOCCER_REQUIRE_POSTGRES"), "true");
        assert_eq!(value_for("SOCCER_EVOLUTION_ENABLED"), "true");
        assert_eq!(
            value_for("SOCCER_POSTGRES_COMPLETED_RUN_RETENTION_GAMES"),
            "200"
        );
        assert_eq!(value_for("SOCCER_SOURCE_COMMIT"), "abc123");
        assert!(env.iter().any(
            |entry| entry["name"] == "SOCCER_DATABASE_URL" && entry.get("valueFrom").is_some()
        ));
    }

    #[test]
    fn missing_required_args_are_rejected() {
        let error = parse_args_from("renderer", ["--run-id", "run-a"]).expect_err("missing arg");
        assert!(error.to_string().contains("--source-sha256 is required"));
    }
}
