#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Test runner for the Fraud Detection System
# Usage: bash scripts/run_tests.sh [--coverage] [--verbose] [--unit-only]
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

COVERAGE=false
VERBOSE=false
UNIT_ONLY=false

for arg in "$@"; do
    case $arg in
        --coverage)   COVERAGE=true ;;
        --verbose)    VERBOSE=true ;;
        --unit-only)  UNIT_ONLY=true ;;
        *)            echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/dashboard:$PROJECT_ROOT:$PYTHONPATH"

echo "=================================================="
echo "  Fraud Detection System - Test Suite"
echo "=================================================="

PYTEST_ARGS=("tests/")

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v")
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS+=("--cov=src" "--cov=dashboard" "--cov-report=term-missing" "--cov-report=html:htmlcov")
fi

if [ "$UNIT_ONLY" = true ]; then
    # Skip tests that need Spark (quality_gates) for faster feedback
    PYTEST_ARGS+=("--ignore=tests/test_quality_gates.py")
fi

echo "Running: pytest ${PYTEST_ARGS[*]}"
echo ""

python -m pytest "${PYTEST_ARGS[@]}"

echo ""
echo "=================================================="
echo "  All tests passed!"
echo "=================================================="
