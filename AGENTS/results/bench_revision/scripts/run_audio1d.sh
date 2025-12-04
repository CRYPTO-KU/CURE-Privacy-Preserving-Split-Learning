#!/bin/bash
# Run PTB-XL CNN (audio1d) benchmarks for prior work comparison
# Cores: 6 (Khan et al.)

set -e  # Exit on error

LOG_N=13
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BENCHMARK_EXEC="benchlayer_linux"  # Expected to be in current directory or PATH

echo "=========================================="
echo "PTB-XL CNN (audio1d) Benchmark - Prior Work Comparison"
echo "=========================================="

# Check if benchmark executable exists (in current directory or PATH)
if [ ! -f "./$BENCHMARK_EXEC" ] && ! command -v "$BENCHMARK_EXEC" &> /dev/null; then
    echo "ERROR: Benchmark executable '$BENCHMARK_EXEC' not found."
    echo "Please build it first:"
    echo "  GOOS=linux GOARCH=amd64 go build -o benchlayer_linux cmd/benchmarks/main.go"
    exit 1
fi

# Use ./benchlayer_linux if it exists in current directory, otherwise use PATH version
if [ -f "./$BENCHMARK_EXEC" ]; then
    BENCHMARK_EXEC="./$BENCHMARK_EXEC"
fi

# Run benchmark for 6 cores
CORES=6
echo ""
echo "Running benchmark with $CORES cores, logN=$LOG_N..."
OUTPUT_FILE="bench_results_cores${CORES}_logn${LOG_N}.csv"

if [ -f "$OUTPUT_FILE" ]; then
    echo "WARNING: $OUTPUT_FILE already exists. Skipping..."
else
    "$BENCHMARK_EXEC" -cores "$CORES" -logn "$LOG_N" > "bench_${CORES}cores_${LOG_N}logn.log" 2>&1
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✓ Successfully created $OUTPUT_FILE"
    else
        echo "✗ ERROR: $OUTPUT_FILE was not created!"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "PTB-XL CNN benchmarks completed!"
echo "=========================================="
echo "Output files:"
ls -lh bench_results_cores*_logn${LOG_N}.csv 2>/dev/null || echo "No CSV files found"
