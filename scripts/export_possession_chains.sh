#!/usr/bin/env bash
# Wrapper for the possession-chain EPV export (docs/how-to-climb-codex-conversation.md, Step 2).
# Builds (if needed) and runs soccer_export_possession_chains, then fits an EPV grid.
#   scripts/export_possession_chains.sh [games=40] [minutes=6] [out=/tmp/epv-export/chains.jsonl] [--features]
set -uo pipefail
cd "$(dirname "$0")/.."
GAMES="${1:-40}"; MINUTES="${2:-6}"; OUT="${3:-/tmp/epv-export/chains.jsonl}"
FEAT=""; [[ " $* " == *" --features "* ]] && FEAT="--features"
TARGET="${CARGO_TARGET_DIR:-/tmp/epv-export-target}"
BIN="$TARGET/release/soccer_export_possession_chains"
mkdir -p "$(dirname "$OUT")"
CARGO_TARGET_DIR="$TARGET" nice -n 12 cargo build --release --bin soccer_export_possession_chains >&2
nice -n 12 "$BIN" "$OUT" "$GAMES" "$MINUTES" EP7C0000 $FEAT
python3 scripts/fit_epv_grid.py "$OUT" "${OUT%.jsonl}_epv_grid.json"
