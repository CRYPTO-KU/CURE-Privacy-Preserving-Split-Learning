#!/bin/bash
# Upload and monitor script for bench_revision experiments
# Usage: bash upload_and_monitor.sh [remote_path]

set -e

REMOTE_USER="hkanpak"
REMOTE_HOST="morpheus.cs.bilkent.edu.tr"
REMOTE_PATH="${1:-~/bench_revision}"  # Default to ~/bench_revision if not provided

# Get the script directory and find CURE_lib root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CURE_lib root is 4 levels up from AGENTS/results/bench_revision/
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LOCAL_DIR="$SCRIPT_DIR"
BENCHMARK_EXEC="benchlayer_linux"

echo "=========================================="
echo "Upload and Monitor Bench Revision Experiments"
echo "=========================================="
echo "Remote destination: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "Repository root: $REPO_ROOT"
echo ""

# Step 1: Check if benchmark executable exists locally (in repo root or current dir)
if [ ! -f "$REPO_ROOT/$BENCHMARK_EXEC" ] && [ ! -f "$SCRIPT_DIR/$BENCHMARK_EXEC" ]; then
    echo "⚠️  Benchmark executable not found. Building it..."
    echo "Building benchmark executable..."
    cd "$REPO_ROOT"
    GOOS=linux GOARCH=amd64 go build -o "$BENCHMARK_EXEC" cmd/benchmarks/main.go
    
    if [ ! -f "$REPO_ROOT/$BENCHMARK_EXEC" ]; then
        echo "❌ Failed to build benchmark executable!"
        exit 1
    fi
    echo "✓ Built successfully"
    BENCHMARK_PATH="$REPO_ROOT/$BENCHMARK_EXEC"
else
    if [ -f "$REPO_ROOT/$BENCHMARK_EXEC" ]; then
        BENCHMARK_PATH="$REPO_ROOT/$BENCHMARK_EXEC"
        echo "✓ Benchmark executable found in repo root"
    else
        BENCHMARK_PATH="$SCRIPT_DIR/$BENCHMARK_EXEC"
        echo "✓ Benchmark executable found in script directory"
    fi
fi

# Step 2: Upload directory
echo ""
echo "Step 1: Uploading bench_revision directory..."
echo "----------------------------------------"
rsync -avz --progress "$LOCAL_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

# Step 3: Upload benchmark executable
echo ""
echo "Step 2: Uploading benchmark executable..."
echo "----------------------------------------"
rsync -avz --progress "$BENCHMARK_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/benchlayer_linux"

echo ""
echo "=========================================="
echo "Upload Complete!"
echo "=========================================="
echo ""
echo "To SSH and run experiments:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_PATH}"
echo "  bash run_all_experiments.sh"
echo ""
echo "To monitor progress in real-time:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_PATH}"
echo "  tail -f bench_*cores_*logn.log"
echo ""
echo "To run in background with screen:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  screen -S bench_experiments"
echo "  cd ${REMOTE_PATH}"
echo "  bash run_all_experiments.sh"
echo "  # Press Ctrl+A then D to detach"
echo "  # Reattach with: screen -r bench_experiments"
