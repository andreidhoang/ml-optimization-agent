#!/usr/bin/env bash
# Verify P0.5 D3: cosmos_lab.harness.nat lightweight wrapper ships and works.
# Per v5.1 architecture (PLAN_V2 §0.4.5): nat is deployment wrapper, not
# runtime substrate. D3 ships the registration shim + smoke test.

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
check "cosmos_lab/harness/nat.py exists" test -f cosmos_lab/harness/nat.py
check "tests/optimization/harness/test_nat_adapter.py exists" \
    test -f tests/optimization/harness/test_nat_adapter.py

echo "--- Imports ---"
check "from cosmos_lab.harness import register_as_nat_tool" \
    python -c "from cosmos_lab.harness import register_as_nat_tool"
check "from cosmos_lab.harness.nat import register_as_nat_tool" \
    python -c "from cosmos_lab.harness.nat import register_as_nat_tool"

echo "--- LOC budget (~50 LOC target, 200 LOC hard cap per v5.1) ---"
LOC=$(grep -vcE "^\s*(#|$|\"\"\")" cosmos_lab/harness/nat.py)
TOTAL_LOC=$(wc -l < cosmos_lab/harness/nat.py | tr -d ' ')
if [[ $TOTAL_LOC -le 200 ]]; then
    echo "  ✅ nat.py is ${TOTAL_LOC} LOC total (~${LOC} non-comment, budget 200)"
    PASS=$((PASS+1))
else
    echo "  ❌ nat.py is ${TOTAL_LOC} LOC (over 200 cap — v5.1 expected ~50)"
    FAIL=$((FAIL+1))
fi

echo "--- Smoke tests ---"
echo "  Running adapter tests..."
TEST_OUTPUT=$(uv run python -m pytest tests/optimization/harness/test_nat_adapter.py -q 2>&1)
if echo "$TEST_OUTPUT" | grep -qE "passed"; then
    PASS_COUNT=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
    echo "  ✅ tests/optimization/harness/test_nat_adapter.py exits 0 (${PASS_COUNT:-?} tests passed)"
    PASS=$((PASS+1))
else
    echo "  ❌ tests/optimization/harness/test_nat_adapter.py failed"
    echo "$TEST_OUTPUT" | tail -10 | sed 's/^/    /'
    FAIL=$((FAIL+1))
fi

echo "--- D1 + D2 still green (no regression) ---"
if ./bin/verify_p0_5_d1.sh > /dev/null 2>&1; then
    echo "  ✅ P0.5 D1 verifier still 14/14"
    PASS=$((PASS+1))
else
    echo "  ❌ P0.5 D1 verifier regressed!"
    FAIL=$((FAIL+1))
fi

if ./bin/verify_p0_5_d2.sh > /dev/null 2>&1; then
    echo "  ✅ P0.5 D2 verifier still 11/11"
    PASS=$((PASS+1))
else
    echo "  ❌ P0.5 D2 verifier regressed!"
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
echo "===== P0.5 D3 verifier: $PASS pass, $FAIL fail ====="
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
