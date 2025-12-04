#!/bin/bash
# =============================================================================
# CURE Artifact - Automated Experiment Runner
# PoPETs 2026 Artifact Evaluation
# =============================================================================
# Paper: CURE: Privacy-Preserving Split Learning Done Right
# Authors: Halil Ibrahim Kanpak, Aqsa Shabbir, Esra Genç, Alptekin Küpçü, Sinem Sav
#
# This script runs all experiments to reproduce the paper's main claims.
# Usage: ./scripts/run_all_experiments.sh [--quick|--full]
#
# Options:
#   --quick   Run simplified experiments (~15 min, for initial validation)
#   --full    Run complete experiments (~1-2 hours, for full reproduction)
#
# Data Requirements:
#   - MNIST dataset: 784-dimensional input vectors
#   - Data can be downloaded from: http://yann.lecun.com/exdb/mnist/
#   - Place in data/mnist/raw/ directory
#   - For timing benchmarks, synthetic data is generated automatically
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default mode
MODE="${1:---quick}"
RESULTS_DIR="artifact_results_$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}CURE Artifact - Automated Experiment Runner${NC}"
echo -e "${BLUE}Paper: Privacy-Preserving Split Learning Done Right${NC}"
echo -e "${BLUE}Mode: ${MODE}${NC}"
echo -e "${BLUE}Results directory: ${RESULTS_DIR}${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

cd "$ROOT_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Experiment 1: Layer Correctness Tests (Paper Section 5.1)
# Claim C1: HE-encrypted layers produce numerically correct results
# =============================================================================
echo -e "${YELLOW}[Experiment 1/4] Layer Correctness Tests (Paper §5.1)${NC}"
echo "Purpose: Validate HE layers match plaintext within RMS error < 1e-4"
echo "Claim: C1 - HE layers are numerically correct"
echo ""

echo "Running correctness tests..."
if go test ./nn/... -v -run Correctness 2>&1 | tee "$RESULTS_DIR/exp1_correctness.log"; then
    echo -e "${GREEN}✓ Experiment 1 PASSED: Layer correctness verified${NC}"
else
    echo -e "${RED}✗ Experiment 1 FAILED: Check $RESULTS_DIR/exp1_correctness.log${NC}"
fi
echo ""

# =============================================================================
# Experiment 2: Unit Tests (Paper Section 4)
# Claim: All system components function correctly
# =============================================================================
echo -e "${YELLOW}[Experiment 2/4] Unit Tests (Paper §4)${NC}"
echo "Purpose: Verify all system components work correctly"
echo ""

echo "Running unit tests..."
if go test ./... -v -short 2>&1 | tee "$RESULTS_DIR/exp2_unit_tests.log"; then
    echo -e "${GREEN}✓ Experiment 2 PASSED: All unit tests passed${NC}"
else
    echo -e "${YELLOW}! Experiment 2: Some tests may have failed - check log${NC}"
fi
echo ""

# =============================================================================
# Experiment 3: Performance Benchmarks (Paper Section 6, Tables 2-4)
# Claims C2, C3: HE operations achieve practical performance
# =============================================================================
echo -e "${YELLOW}[Experiment 3/4] Performance Benchmarks (Paper §6, Tables 2-4)${NC}"
echo "Purpose: Measure layer-by-layer HE performance"
echo "Claims: C2 - Split learning with HE is practical"
echo "        C3 - Parallelization improves performance"
echo ""
echo "Note: Benchmarks use synthetic data (784-dim vectors for MNIST-like input)"
echo ""

cd cmd/benchmarks

# Build benchmark binary
echo "Building benchmark binary..."
go build -o benchmark .

if [ "$MODE" == "--quick" ]; then
    echo "Running QUICK benchmarks (1 core, 3 iterations)..."
    ./benchmark --logN=13 --cores=1 --iterations=3 2>&1 | tee "../../$RESULTS_DIR/exp3_bench_quick.log"
    cp bench_results_*.csv "../../$RESULTS_DIR/" 2>/dev/null || true
else
    echo "Running FULL benchmarks (multiple cores, 5 iterations each)..."
    for cores in 1 2 4 8; do
        echo "  Running with $cores core(s)..."
        ./benchmark --logN=13 --cores=$cores --iterations=5 2>&1 | tee "../../$RESULTS_DIR/exp3_bench_cores${cores}.log"
    done
    cp bench_results_*.csv "../../$RESULTS_DIR/" 2>/dev/null || true
fi

cd ../..

echo -e "${GREEN}✓ Experiment 3 COMPLETED: Results saved to $RESULTS_DIR/${NC}"
echo ""

# =============================================================================
# Experiment 4: Results Comparison (Paper Tables 2-4)
# Claim: Results reproducible within 5% of reported values
# =============================================================================
echo -e "${YELLOW}[Experiment 4/4] Results Comparison${NC}"
echo "Purpose: Compare new results with paper's reported values"
echo "Tolerance: 5% for same hardware, 15% for different hardware"
echo ""

# Reference results from paper
REF_RESULTS="results/reference/bench_results_cores4_logn13.csv"
NEW_RESULTS="$RESULTS_DIR/bench_results_cores1_logn13.csv"

if [ -f "$REF_RESULTS" ]; then
    echo "Reference results: $REF_RESULTS"
    
    if [ -f "$NEW_RESULTS" ]; then
        echo "New results: $NEW_RESULTS"
        echo ""
        
        # Run comparison script
        if python3 scripts/compare_results.py --new "$NEW_RESULTS" --ref "$REF_RESULTS" --tolerance 0.15 2>&1 | tee "$RESULTS_DIR/exp4_comparison.log"; then
            echo -e "${GREEN}✓ Experiment 4: Results comparison completed${NC}"
        else
            echo -e "${YELLOW}! Experiment 4: Some differences detected (may be hardware-related)${NC}"
        fi
    else
        echo -e "${YELLOW}! New results file not found: $NEW_RESULTS${NC}"
        echo "  Run benchmarks first to generate results."
    fi
else
    echo -e "${YELLOW}! Reference results not found: $REF_RESULTS${NC}"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}EXPERIMENT SUMMARY${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo "All results saved to: $RESULTS_DIR/"
echo ""
echo "Files generated:"
ls -la "$RESULTS_DIR/" 2>/dev/null || echo "  (no files yet)"
echo ""
echo -e "${GREEN}Paper Claims Mapping:${NC}"
echo "  C1 (Layer Correctness)     -> exp1_correctness.log    [Paper §5.1]"
echo "  C2 (Practical Performance) -> exp3_bench_*.log        [Paper §6, Tables 2-4]"
echo "  C3 (Parallelization)       -> Compare cores1 vs cores8 [Paper §6.3]"
echo ""
echo -e "${YELLOW}Reproducibility Notes:${NC}"
echo "  - Paper hardware: 2× Intel Xeon E5-2650 v3 @ 2.30GHz (40 threads)"
echo "  - Same hardware: Results should be within 5%"
echo "  - Different hardware: Focus on relative speedup trends"
echo "  - MNIST input dimension: 784 (28×28 flattened images)"
echo ""
echo -e "${GREEN}Artifact evaluation complete!${NC}"
