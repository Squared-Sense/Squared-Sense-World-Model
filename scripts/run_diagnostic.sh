#!/usr/bin/env bash
# ==============================================================================
# Squared Sense: Fast 10-Second Diagnostic Check Script
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

exec uv run python scratch/diag_check.py "$@"
