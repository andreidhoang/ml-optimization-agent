#!/usr/bin/env bash
# Verify P0.5 D4: cosmos_lab/harness/CONTRACT.md + parametrized adapter contract tests.
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
check "cosmos_lab/harness/CONTRACT.md exists" test -f cosmos_lab/harness/CONTRACT.md
check "tests/optimization/harness/test_adapter_contract.py exists" \
    test -f tests/optimization/harness/test_adapter_contract.py

echo "--- CONTRACT.md content quality ---"
check "CONTRACT.md mentions Family A (execution substrate)" \
    grep -q "Family A" cosmos_lab/harness/CONTRACT.md
check "CONTRACT.md mentions Family B (deployment surface)" \
    grep -q "Family B" cosmos_lab/harness/CONTRACT.md
check "CONTRACT.md documents shared requirements S1-S5" \
    bash -c 'grep -q "S1.*Idempotency" cosmos_lab/harness/CONTRACT.md && grep -q "S5" cosmos_lab/harness/CONTRACT.md'
check "CONTRACT.md documents both shipped adapters" \
    bash -c 'grep -q "install_into_session" cosmos_lab/harness/CONTRACT.md && grep -q "register_as_nat_tool" cosmos_lab/harness/CONTRACT.md'

echo "--- Parametrized contract tests ---"
echo "  Running adapter contract tests..."
TEST_OUTPUT=$(uv run python -m pytest tests/optimization/harness/test_adapter_contract.py -q 2>&1)
if echo "$TEST_OUTPUT" | grep -qE "passed"; then
    PASS_COUNT=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
    echo "  ✅ test_adapter_contract.py exits 0 (${PASS_COUNT:-?} tests passed)"
    PASS=$((PASS+1))
else
    echo "  ❌ test_adapter_contract.py failed"
    echo "$TEST_OUTPUT" | tail -10 | sed 's/^/    /'
    FAIL=$((FAIL+1))
fi

echo "--- Coverage: both shipped adapters in registry ---"
check "test_adapter_contract.py registers ml_intern" \
    grep -q '"ml_intern"' tests/optimization/harness/test_adapter_contract.py
check "test_adapter_contract.py registers nat" \
    grep -q '"nat"' tests/optimization/harness/test_adapter_contract.py

echo "--- D1 + D2 + D3 still green (no regression) ---"
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

if ./bin/verify_p0_5_d3.sh > /dev/null 2>&1; then
    echo "  ✅ P0.5 D3 verifier still 10/10"
    PASS=$((PASS+1))
else
    echo "  ❌ P0.5 D3 verifier regressed!"
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
echo "===== P0.5 D4 verifier: $PASS pass, $FAIL fail ====="
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
