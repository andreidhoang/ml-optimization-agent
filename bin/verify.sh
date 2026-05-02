#!/usr/bin/env bash
# verify.sh — phase verifier router
# Usage: ./bin/verify.sh <phase>
# Returns: 0 = pass, 1 = fail
# Phases: p0_5_d1, p0_5, p1, p2, ... (add as you go)

set -euo pipefail

PHASE="${1:-}"

if [[ -z "$PHASE" ]]; then
    echo "Usage: $0 <phase>"
    echo "Phases:"
    ls bin/verify_*.sh 2>/dev/null | sed 's|bin/verify_|  |;s|\.sh||'
    exit 1
fi

SCRIPT="bin/verify_${PHASE}.sh"
if [[ ! -f "$SCRIPT" ]]; then
    echo "❌ No verifier for phase '$PHASE'"
    echo "Expected: $SCRIPT"
    exit 1
fi

echo "=== Verifying phase: $PHASE ==="
exec bash "$SCRIPT"
