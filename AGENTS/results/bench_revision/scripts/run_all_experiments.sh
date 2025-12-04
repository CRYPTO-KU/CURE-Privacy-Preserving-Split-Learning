#!/bin/bash
# Main script to run all prior work comparison experiments

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Prior Work Comparison Experiments"
echo "=========================================="
echo "This script will run benchmarks for:"
echo "  - MNIST MLP (mnistfc): 24 cores (Glyph), 30 cores (Nandakumar)"
echo "  - PTB-XL CNN (audio1d): 6 cores (Khan)"
echo ""
echo "Note: ResNet excluded (requires 112 cores)"
echo "=========================================="

# Check if benchmark executable exists (in current directory or PATH)
BENCHMARK_EXEC="benchlayer_linux"
if [ ! -f "./$BENCHMARK_EXEC" ] && ! command -v "$BENCHMARK_EXEC" &> /dev/null; then
    echo ""
    echo "ERROR: Benchmark executable '$BENCHMARK_EXEC' not found."
    echo ""
    echo "Please build it first:"
    echo "  cd /path/to/CURE_lib"
    echo "  GOOS=linux GOARCH=amd64 go build -o benchlayer_linux cmd/benchmarks/main.go"
    echo ""
    echo "Then copy it to this directory or ensure it's in PATH."
    exit 1
fi

# Use ./benchlayer_linux if it exists in current directory, otherwise use PATH version
if [ -f "./$BENCHMARK_EXEC" ]; then
    BENCHMARK_EXEC="./$BENCHMARK_EXEC"
fi

# Run MNIST FC benchmarks
echo ""
echo "Step 1/2: Running MNIST FC benchmarks..."
echo "----------------------------------------"
bash run_mnistfc.sh

# Run Audio1D benchmarks
echo ""
echo "Step 2/2: Running PTB-XL CNN (audio1d) benchmarks..."
echo "----------------------------------------"
bash run_audio1d.sh

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the generated CSV files:"
echo "     ls -lh bench_results_cores*_logn13.csv"
echo ""
echo "  2. Aggregate results into cut timings:"
echo "     bash aggregate_results.sh"
echo ""
echo "  3. The aggregated results will be in cut_aggregates.csv"
echo ""
