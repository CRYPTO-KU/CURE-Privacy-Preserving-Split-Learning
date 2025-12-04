# Artifact Appendix

**Paper Title**: CURE: Privacy-Preserving Split Learning Done Right

**Authors**: Halil Ibrahim Kanpak, Aqsa Shabbir, Esra Genç, Alptekin Küpçü, Sinem Sav

**Requested Badges**: Available, Functional, Reproduced

---

## Description

CURE is a Go library for privacy-preserving split learning using Homomorphic Encryption (HE) with the CKKS scheme via Lattigo v6. It provides encrypted neural network layers (Linear, Activation, Conv2D, AvgPool2D), split learning protocols, and comprehensive benchmarking tools. CURE enables secure split learning while substantially improving communication and parallelization compared to prior work.

### Security/Privacy Issues and Ethical Concerns

This artifact does not contain any security vulnerabilities, malware, or personally identifiable information. The MNIST dataset used for evaluation is publicly available. There are no ethical concerns with making this artifact publicly available.

---

## Basic Requirements

### Hardware Requirements

- **Minimum**: 8 GB RAM, 4-core CPU
- **Recommended**: 16+ GB RAM, 8+ cores for parallel benchmarks
- **Storage**: ~500 MB for source code, dependencies, and benchmark results

The artifact is CPU-intensive due to HE operations. While no specialized hardware (GPU, TEE, etc.) is required, performance scales with available CPU cores.

**Evaluation Hardware (Paper Results)**:
| Component | Specification |
|-----------|---------------|
| CPU | 2× Intel Xeon E5-2650 v3 @ 2.30GHz |
| Cores | 20 physical cores (40 threads with HT) |
| RAM | 251 GB |
| OS | Ubuntu 18.04.6 LTS |
| GPU | Not used (CPU-only HE operations) |

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS 12+
- **Go**: Version 1.23 or higher
- **Python 3**: For result comparison scripts
- **Docker**: Optional, for containerized execution (Docker 20.10+)
- **Git**: For cloning the repository

All Go dependencies (including Lattigo v6) are managed via Go modules and will be downloaded automatically.

### Estimated Time and Storage Consumption

| Task | Time | Storage |
|------|------|---------|
| Clone repository | ~1 min | ~100 MB |
| Download dependencies | ~2 min | ~300 MB |
| Build all binaries | ~1 min | ~50 MB |
| Run unit tests | ~5-10 min | minimal |
| Run quick benchmarks (logN=13, 1 core) | ~10-15 min | ~10 MB |
| Run full benchmarks (all cores) | ~1-2 hours | ~50 MB |

---

## Environment

### Accessibility

The artifact is publicly available on GitHub:
- **Repository**: https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning
- **Version**: Tag `v1.0-popets` (Commit: `35f85ae`)
- **License**: GNU General Public License v3.0 (GPL-3.0)

### Set up the environment

#### Option 1: Native Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning.git
cd CURE-Privacy-Preserving-Split-Learning

# Checkout the artifact version
git checkout v1.0-popets

# Download Go dependencies
go mod download

# Build all binaries
make build
```

#### Option 2: Docker

```bash
# Clone the repository
git clone https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning.git
cd CURE-Privacy-Preserving-Split-Learning
git checkout v1.0-popets

# Build Docker image (uses pinned versions: Go 1.23.3, Alpine 3.19.1)
docker build -t cure:latest .

# Run interactive shell
docker run -it cure:latest /bin/sh
```

### Testing the Environment

Verify the installation:

```bash
# Run unit tests
make test

# Or run a quick smoke test
go test ./core/ckkswrapper/... -v -short
go test ./nn/layers/... -v -short
```

Expected output: All tests should pass with `PASS` status.

### Data Requirements

**MNIST Dataset**:
- Input dimension: **784** (28×28 flattened images)
- Download from: http://yann.lecun.com/exdb/mnist/
- Place raw files in `data/mnist/raw/`
- **Note**: For timing benchmarks, synthetic data is generated automatically. Real MNIST data is only needed for accuracy experiments.

---

## Artifact Evaluation

### One-Click Reproduction (Recommended)

For minimal reviewer effort, use the automated experiment script:

```bash
# Quick validation (~15 minutes)
./scripts/run_all_experiments.sh --quick

# Full reproduction (~1-2 hours)
./scripts/run_all_experiments.sh --full
```

This script runs all experiments and saves results to `artifact_results_<timestamp>/`.

---

### Main Results and Claims

| Claim | Paper Section | Experiment | Expected Result |
|-------|---------------|------------|-----------------|
| **C1**: HE layers are numerically correct | §5.1 | Exp 1 | RMS error < 1e-4 |
| **C2**: Split learning with HE is practical | §6, Tables 2-4 | Exp 2-3 | Timings within 5% |
| **C3**: Parallelization improves performance | §6.3 | Exp 3 | Near-linear speedup |

### Experiments

#### Experiment 1: Layer Correctness Tests (Claim C1)

**Purpose**: Validate that HE-encrypted layers produce numerically correct results compared to plaintext (Paper §5.1).

**Commands**:
```bash
# Run all correctness tests
go test ./nn/... -v -run Correctness

# Run specific layer tests
go test ./nn/layers/... -v -run TestLinear
go test ./nn/layers/... -v -run TestConv
go test ./nn/layers/... -v -run TestActivation
go test ./nn/layers/... -v -run TestAvgPool
```

**Expected results**: All tests pass. HE results match plaintext within RMS error < 1e-4.

**Time**: ~5 minutes

---

#### Experiment 2: Quick Benchmark - Simplified (Claims C2, C3)

**Purpose**: Reproduce benchmark results with reduced parameters for faster evaluation.

**Note**: Uses synthetic 784-dimensional input vectors (MNIST-like).

**Commands**:
```bash
cd cmd/benchmarks
go build -o benchmark .

# Run quick benchmark (single core, logN=13, 3 iterations)
./benchmark --logN=13 --cores=1 --iterations=3
```

**Expected results**: 
- CSV file `bench_results_cores1_logn13.csv` with layer-by-layer timing
- Forward pass times: 10-500ms depending on layer type

**Time**: ~10-15 minutes

---

#### Experiment 3: Full Benchmark Suite (Claims C2, C3)

**Purpose**: Reproduce the complete benchmark results from the paper (Tables 2-4).

**Commands**:
```bash
cd cmd/benchmarks

# Run benchmarks with different core counts
for cores in 1 2 4 8; do
    ./benchmark --logN=13 --cores=$cores --iterations=5
done
```

**Expected results**: 
- CSV files: `bench_results_cores{1,2,4,8}_logn13.csv`
- Timing results within 5% of paper values (same hardware) or 15% (different hardware)
- Speedup ratio: ~1.8-2× when doubling cores

**Time**: ~1-2 hours for complete sweep

---

#### Experiment 4: Model Inference Benchmarks (Claim C2)

**Purpose**: Measure end-to-end inference time for different model architectures.

**Commands**:
```bash
make build

# Run inference benchmark for different models
./bin/cure-bench --model=mnist --logN=13 --cores=4
./bin/cure-bench --model=lenet --logN=13 --cores=4
./bin/cure-bench --model=bcw --logN=13 --cores=4
```

**Expected results**: Per-model inference times matching Table 3 in the paper.

**Time**: ~30 minutes per model

---

### Reproducing Paper Results

Pre-computed paper results are included for comparison:
- `results/reference/bench_results_cores4_logn13.csv` - Reference benchmark results

**Automated Comparison** (Recommended):
```bash
# Compare your results with reference (15% tolerance for different hardware)
python3 scripts/compare_results.py \
    --new bench_results_cores4_logn13.csv \
    --ref results/reference/bench_results_cores4_logn13.csv \
    --tolerance 0.15
```

The script outputs:
- Per-metric comparison with pass/fail status
- Overall reproducibility assessment
- Detailed breakdown of any differences

**Interpretation**:
- **Same hardware**: Results within 5% of reported values
- **Different hardware**: Focus on relative speedup ratios and trends

---

## Limitations

1. **Hardware Variability**: Absolute benchmark times depend on CPU performance. Results should be compared relatively, not absolutely.

2. **Memory Requirements**: Large ring sizes (logN ≥ 15) require 16+ GB RAM. Use logN=13 for resource-constrained environments.

3. **Long-Running Experiments**: Full benchmark sweeps take 1-2 hours. Use `--quick` mode for faster validation.

4. **Dataset**: MNIST processed binary files (>100MB) are excluded from the repository. For timing benchmarks, synthetic 784-dimensional data is generated automatically. For accuracy experiments, download MNIST from http://yann.lecun.com/exdb/mnist/.

---

## Notes on Reusability

This artifact is designed for reuse and extension:

1. **Adding New Layers**: Implement the `nn.Module` interface in `nn/layers/`
2. **Custom Models**: Define architectures in `nn/bench/models.go`
3. **Integration**: Import as a Go module: `import "cure_lib/core/ckkswrapper"`

See `examples/README.md` for integration examples.

---

## Badge Checklists

### Available Badge ✓
- [x] Publicly available artifact: https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning
- [x] Persistent link: Tag `v1.0-popets` (Commit: `35f85ae`)
- [x] License present: GPL-3.0
- [x] Relevant to paper: Implements all described HE layers and benchmarks

### Functional Badge ✓
- [x] Clear documentation: README.md + ARTIFACT-APPENDIX.md
- [x] Completeness: All layers, benchmarks, and split learning code included
- [x] Exercisability: Dockerfile + Makefile with pinned versions (Go 1.23.3, Alpine 3.19.1)

### Reproduced Badge ✓
- [x] Claims identified: C1 (Correctness), C2 (Performance), C3 (Scalability)
- [x] Claims → Experiments mapping: Table provided above
- [x] Automation: `./scripts/run_all_experiments.sh`
- [x] Automated comparison: `python3 scripts/compare_results.py`
- [x] Expected results: Pre-computed results in `results/reference/`

---

## Version

This artifact appendix follows the PoPETs 2026 Artifact Evaluation guidelines.
