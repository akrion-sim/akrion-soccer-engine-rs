#!/usr/bin/env bash
# Prune stale local Cargo artifacts that this repo can recreate before the next build.
# The deletion is intentionally narrow: guarded target dirs only, known rustc/Cargo
# artifact names only, and an age gate to avoid racing an active build.

set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${CARGO_TARGET_DIR:-$ROOT/target}}"
AGE_SECONDS="${SOCCER_LOCAL_ARTIFACT_PRUNE_AGE_SECONDS:-3600}"
DRY_RUN="${SOCCER_LOCAL_ARTIFACT_PRUNE_DRY_RUN:-0}"

case "${SOCCER_SKIP_LOCAL_ARTIFACT_PRUNE:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    printf 'soccer local artifact prune skipped by SOCCER_SKIP_LOCAL_ARTIFACT_PRUNE\n' >&2
    exit 0
    ;;
esac

ROOT="$ROOT" TARGET_DIR="$TARGET_DIR" AGE_SECONDS="$AGE_SECONDS" DRY_RUN="$DRY_RUN" python3 - <<'PY'
import fcntl
import os
import shutil
import sys
import time
from pathlib import Path

root = Path(os.environ["ROOT"]).resolve()
target = Path(os.environ["TARGET_DIR"]).expanduser().resolve()

try:
    age_seconds = int(os.environ.get("AGE_SECONDS", "3600").strip() or "3600")
except ValueError:
    print("soccer local artifact prune: AGE_SECONDS must be an integer", file=sys.stderr)
    sys.exit(2)

if age_seconds < 60:
    print("soccer local artifact prune: refusing age gate below 60 seconds", file=sys.stderr)
    sys.exit(2)

dry_raw = os.environ.get("DRY_RUN", "0").strip().lower()
dry_run = dry_raw in {"1", "true", "yes", "on"}

target_like = target.name == "target" or target.name.endswith("-target") or "target" in target.name
allowed = (
    target == root / "target"
    or (root in target.parents and target_like)
    or (root / "tmp") in target.parents
    or str(target).startswith("/tmp/")
    or str(target).startswith("/private/tmp/")
)
if not allowed or not target_like:
    print(f"soccer local artifact prune: refusing unguarded target dir {target}", file=sys.stderr)
    sys.exit(2)

if not target.exists():
    print(f"soccer local artifact prune: target dir does not exist: {target}", file=sys.stderr)
    sys.exit(0)

cutoff = time.time() - age_seconds
deleted_files = 0
deleted_bytes = 0
skipped_recent = 0
errors = 0
active_profiles = 0

def acquire_profile_lock(profile: Path):
    lock_path = profile / ".cargo-lock"
    try:
        handle = lock_path.open("a+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except (BlockingIOError, OSError):
        try:
            handle.close()
        except (NameError, OSError):
            pass
        return None

def release_profile_lock(handle) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()

def tree_stats(path: Path) -> tuple[int, int, float]:
    files = 0
    size = 0
    newest = 0.0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        files += 1
        size += stat.st_size
        newest = max(newest, stat.st_mtime)
    return files, size, newest

incremental_raw = os.environ.get("CARGO_INCREMENTAL", "0").strip().lower()
incremental_enabled = incremental_raw in {"1", "true", "yes", "on"}
if not incremental_enabled:
    for path in sorted(target.rglob("incremental")):
        if not path.is_dir():
            continue
        profile_lock = acquire_profile_lock(path.parent)
        if profile_lock is None:
            active_profiles += 1
            continue
        try:
            files, size, newest = tree_stats(path)
            if newest > cutoff:
                skipped_recent += files
                continue
            if dry_run:
                deleted_files += files
                deleted_bytes += size
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                errors += 1
                print(f"soccer local artifact prune: delete failed {path}: {exc}", file=sys.stderr)
                continue
            deleted_files += files
            deleted_bytes += size
        finally:
            release_profile_lock(profile_lock)

# `*.rcgu.o` is safe only after it has aged past any plausible live linker.
# Do not delete hashed rlib/rmeta outputs: an old feature fingerprint may still
# legitimately reference those, and age alone does not prove obsolescence.
rcgu_by_profile = {}
for path in target.rglob("*.rcgu.o"):
    if path.is_file() and path.parent.name == "deps":
        rcgu_by_profile.setdefault(path.parent.parent, []).append(path)

for profile, paths in rcgu_by_profile.items():
    profile_lock = acquire_profile_lock(profile)
    if profile_lock is None:
        active_profiles += 1
        continue
    try:
        for path in paths:
            try:
                stat = path.stat()
            except OSError as exc:
                errors += 1
                print(f"soccer local artifact prune: stat failed {path}: {exc}", file=sys.stderr)
                continue
            if stat.st_mtime > cutoff:
                skipped_recent += 1
                continue
            if dry_run:
                deleted_files += 1
                deleted_bytes += stat.st_size
                continue
            try:
                path.unlink()
            except OSError as exc:
                errors += 1
                print(f"soccer local artifact prune: delete failed {path}: {exc}", file=sys.stderr)
                continue
            deleted_files += 1
            deleted_bytes += stat.st_size
    finally:
        release_profile_lock(profile_lock)

mode = "would_delete" if dry_run else "deleted"
print(
    "soccer local artifact prune: "
    f"{mode}_files={deleted_files} {mode}_bytes={deleted_bytes} "
    f"skipped_recent={skipped_recent} active_profiles={active_profiles} target={target}"
)
if errors:
    sys.exit(1)
PY
