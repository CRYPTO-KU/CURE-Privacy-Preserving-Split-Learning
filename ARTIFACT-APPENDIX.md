# Artifact Appendix

Paper title: **CURE: Privacy-Preserving Split Learning with Homomorphic Encryption**

Artifacts HotCRP Id: (to be filled by authors)

Requested Badge: **Available**, **Functional**, **Reproduced**

## Description

CURE_lib is a Go library for privacy-preserving deep learning using Homomorphic Encryption (HE) with the CKKS scheme via Lattigo v6. It provides encrypted neural network layers (Linear, Activation, Conv2D, AvgPool2D), split learning protocols, and comprehensive benchmarking tools to evaluate HE-based inference and training performance.

### Security/Privacy Issues and Ethical Concerns

This artifact does not contain any security vulnerabilities, malware, or personally identifiable information. The MNIST dataset used for evaluation is publicly available. There are no ethical concerns with making this artifact publicly available.

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

## Environment

### Accessibility

The artifact is publicly available on GitHub:
- **Repository**: https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning
- **Branch**: `main`
- **License**: GNU General Public License v3.0 (GPL-3.0)

### Set up the environment

#### Option 1: Native Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning.git
cd CURE-Privacy-Preserving-Split-Learning

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

# Build Docker image
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

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: HE-Based Neural Network Layer Implementations

The artifact provides correct implementations of HE-encrypted neural network layers that produce results within acceptable numerical precision of plaintext counterparts.

**Relevant sections**: Section 4 (System Design), Section 5.1 (Layer Implementations)

**Experiments**: Unit tests in `nn/layers/*_test.go` validate correctness against plaintext implementations.

#### Main Result 2: Split Learning Performance Benchmarks

The artifact demonstrates practical performance of split learning with HE, measuring forward/backward pass latencies across different model architectures and parallelization settings.

**Relevant sections**: Section 6 (Evaluation), Tables 2-4

**Experiments**: Benchmark suite in `cmd/benchmarks/` with results in `AGENTS/results/`

#### Main Result 3: Scalability with Parallelization

The artifact shows that HE operations benefit from multi-core parallelization, with near-linear speedup for independent operations.

**Relevant sections**: Section 6.3 (Parallelization Analysis)

**Experiments**: Core sweep benchmarks comparing 1, 2, 4, 8, 16, 32 cores

### Experiments

#### Experiment 1: Layer Correctness Tests

**Purpose**: Validate that HE-encrypted layers produce numerically correct results compared to plaintext.

**How to run**:
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

#### Experiment 2: Quick Benchmark (Simplified)

**Purpose**: Reproduce benchmark results with reduced parameters for faster evaluation.

**How to run**:
```bash
cd cmd/benchmarks

# Build benchmark binary
go build -o benchmark .

# Run quick benchmark (single core, logN=13)
./benchmark --logN=13 --cores=1 --iterations=3

# Results will be saved to bench_results_cores1_logn13.csv
```

**Expected results**: CSV file with layer-by-layer timing results. Forward pass times should be in the range of 10-500ms depending on layer type.

**Time**: ~10-15 minutes

#### Experiment 3: Full Benchmark Suite

**Purpose**: Reproduce the complete benchmark results from the paper.

**How to run**:
```bash
cd cmd/benchmarks

# Run benchmarks with different core counts
for cores in 1 2 4 8; do
    ./benchmark --logN=13 --cores=$cores --iterations=5
done

# Results are saved to bench_results_cores{N}_logn13.csv
```

**Expected results**: 
- Benchmark CSV files for each core configuration
- Timing results within 10% of paper-reported values (accounting for hardware differences)
- Speedup ratios consistent with paper's parallelization analysis

**Time**: ~1-2 hours for complete sweep

#### Experiment 4: Model Inference Benchmarks

**Purpose**: Measure end-to-end inference time for different model architectures.

**How to run**:
```bash
# Build inference binary
make build

# Run inference benchmark for different models
./bin/cure-bench --model=mnist --logN=13 --cores=4
./bin/cure-bench --model=lenet --logN=13 --cores=4
./bin/cure-bench --model=bcw --logN=13 --cores=4
```

**Expected results**: Per-model inference times matching Table 3 in the paper (within 15% for different hardware).

**Time**: ~30 minutes per model

### Reproducing Paper Results

The benchmark results reported in the paper can be found in:
- `AGENTS/results/bench0803/` - Main benchmark results
- `AGENTS/results/accuracy_results/` - Accuracy evaluation results

To compare your results with paper results:

```bash
# View paper benchmark results
cat AGENTS/results/bench0803/bench_results_cores4_logn13.csv

# Compare with your results
diff bench_results_cores4_logn13.csv AGENTS/results/bench0803/bench_results_cores4_logn13.csv
```

**Note**: Absolute timing values will vary based on hardware. Focus on:
1. Relative speedup ratios between core configurations
2. Layer-to-layer timing relationships
3. Trend consistency with paper figures

## Limitations

1. **Hardware Variability**: Absolute benchmark times depend on CPU performance. Results should be compared relatively, not absolutely.

2. **Memory Requirements**: Large ring sizes (logN ≥ 15) require 16+ GB RAM. Use logN=13 for resource-constrained environments.

3. **Long-Running Experiments**: Full benchmark sweeps take 1-2 hours. Use `--iterations=3` for quicker validation.

4. **Dataset**: MNIST raw data files are included but processed binary files (>100MB) are excluded from the repository. The benchmark suite generates synthetic data for timing measurements.

## Notes on Reusability

This artifact is designed for reuse and extension:

1. **Adding New Layers**: Implement the `nn.Module` interface in `nn/layers/`
2. **Custom Models**: Define architectures in `nn/bench/models.go`
3. **Integration**: Import as a Go module: `import "cure_lib/core/ckkswrapper"`

See `examples/README.md` for integration examples.

## Version

Based on the template (https://github.com/sysartifacts/artifact-template) produced by the Artifact Evaluation Chairs.
