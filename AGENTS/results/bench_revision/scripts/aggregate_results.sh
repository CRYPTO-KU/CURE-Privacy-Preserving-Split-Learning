#!/bin/bash
# Aggregate layer timings into cut aggregates

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Aggregating Results"
echo "=========================================="

# Check if aggregate_cuts.py exists
if [ ! -f "aggregate_cuts.py" ]; then
    echo "ERROR: aggregate_cuts.py not found!"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found!"
    exit 1
fi

# Check if CSV files exist
CSV_COUNT=$(find . -name "bench_results_cores*_logn*.csv" | wc -l)
if [ "$CSV_COUNT" -eq 0 ]; then
    echo "ERROR: No benchmark CSV files found!"
    echo "Please run benchmarks first: bash run_all_experiments.sh"
    exit 1
fi

echo "Found $CSV_COUNT CSV file(s)"
echo ""

# Run aggregation
OUTPUT_FILE="cut_aggregates.csv"
echo "Running aggregation..."
python3 aggregate_cuts.py --root . --out "$OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Successfully created $OUTPUT_FILE"
    echo ""
    echo "Aggregated results summary:"
    echo "  Output file: $OUTPUT_FILE"
    wc -l "$OUTPUT_FILE"
else
    echo ""
    echo "✗ ERROR: $OUTPUT_FILE was not created!"
    exit 1
fi
