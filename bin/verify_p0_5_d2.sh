#!/usr/bin/env bash
# Verify P0.5 D2: cosmos_lab.harness.ml_intern adapter ships and works.
# Per workflow Phase 1 DEFINE — verifier returns pass/fail, not narrative.

set -uo pipefail

PASS=0
FAIL=0

check() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  ✅ $name"
        PASS=$((PASS+1))
    else
        echo "  ❌ $name"
        FAIL=$((FAIL+1))
    fi
}

echo "--- Structure ---"
check "cosmos_lab/harness/__init__.py exists" test -f cosmos_lab/harness/__init__.py
check "cosmos_lab/harness/ml_intern.py exists" test -f cosmos_lab/harness/ml_intern.py
check "tests/optimization/harness/__init__.py exists" test -f tests/optimization/harness/__init__.py
check "tests/optimization/harness/test_ml_intern_adapter.py exists" \
    test -f tests/optimization/harness/test_ml_intern_adapter.py

echo "--- Imports ---"
check "from cosmos_lab.harness import install_into_session" \
    python -c "from cosmos_lab.harness import install_into_session"
check "from cosmos_lab.harness.ml_intern import install_into_session" \
    python -c "from cosmos_lab.harness.ml_intern import install_into_session"

echo "--- Adapter LOC budget (≤ 200 LOC) ---"
LOC=$(wc -l < cosmos_lab/harness/ml_intern.py | tr -d ' ')
if [[ $LOC -le 200 ]]; then
    echo "  ✅ ml_intern.py is ${LOC} LOC (budget 200)"
    PASS=$((PASS+1))
else
    echo "  ❌ ml_intern.py is ${LOC} LOC (budget 200 — over)"
    FAIL=$((FAIL+1))
fi

echo "--- Smoke tests ---"
echo "  Running adapter tests..."
if uv run python -m pytest tests/optimization/harness/ -q 2>&1 | tail -3; then
    PASS_COUNT=$(uv run python -m pytest tests/optimization/harness/ -q 2>&1 | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
    echo "  ✅ tests/optimization/harness/ exits 0 (${PASS_COUNT:-?} tests passed)"
    PASS=$((PASS+1))
else
    echo "  ❌ tests/optimization/harness/ failed"
    FAIL=$((FAIL+1))
fi

echo "--- D1 still green (no regression) ---"
if ./bin/verify_p0_5_d1.sh > /dev/null 2>&1; then
    echo "  ✅ P0.5 D1 verifier still 14/14"
    PASS=$((PASS+1))
else
    echo "  ❌ P0.5 D1 verifier regressed!"
    FAIL=$((FAIL+1))
fi

echo "--- Upstream baseline preserved ---"
UPSTREAM_FAILS=$(uv run python -m pytest tests/unit/ -q 2>&1 | grep -oE "[0-9]+ failed" | head -1 | grep -oE "[0-9]+")
if [[ "${UPSTREAM_FAILS:-0}" -le 3 ]]; then
    echo "  ✅ tests/unit/ shows ${UPSTREAM_FAILS:-0} failures (≤ 3 known-broken baseline)"
    PASS=$((PASS+1))
else
    echo "  ❌ tests/unit/ shows ${UPSTREAM_FAILS} failures (> 3 — regression!)"
    FAIL=$((FAIL+1))
fi

echo "--- Zero-diff invariant ---"
DIFF=$(git diff upstream/main --name-only 2>/dev/null \
    | grep -v "^cosmos_lab/" \
    | grep -v "^agent/optimization/" \
    | grep -v "^configs/optimization" \
    | grep -v "^tests/optimization" \
    | grep -v "^docs/" \
    | grep -v "^bin/" \
    | grep -v "^pyproject.toml" \
    | grep -v "^uv\.lock" \
    | grep -v "^CLAUDE.md" \
    | grep -v "^AGENTS" \
    | grep -v "^PLAN_V2.md$" \
    | grep -v "^PLAN.md$" \
    | grep -v "^SYSTEM.md$" \
    | grep -v "^EVAL_SPEC.md$" \
    | grep -v "^WORKFLOW.md$" \
    | grep -v "^RESEARCH_AHE_ANALYSIS.md$" \
    | grep -v "^AGENTIC_EVAL_SPEC.md$" \
    | grep -v "^agentic_build_workflow" || true)
if [[ -z "$DIFF" ]]; then
    echo "  ✅ git diff upstream/main --name-only shows owned paths only"
    PASS=$((PASS+1))
else
    echo "  ❌ Unexpected upstream diff:"
    echo "$DIFF" | sed 's/^/    /'
    FAIL=$((FAIL+1))
fi

echo
echo "===== P0.5 D2 verifier: $PASS pass, $FAIL fail ====="
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
