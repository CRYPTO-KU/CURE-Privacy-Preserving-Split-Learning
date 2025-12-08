# Artifact Appendix

Paper title: **CURE: Privacy-Preserving Split Learning Done Right**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [ ] **Reproduced**

## Description

1. **Paper**: CURE: Privacy-Preserving Split Learning Done Right. Halil Ibrahim Kanpak, Aqsa Shabbir, Esra Genç, Alptekin Küpçü, Sinem Sav.
2. **Artifact Description**: CURE is a Go library for privacy-preserving split learning using Homomorphic Encryption (HE) with the CKKS scheme via Lattigo v6. It provides encrypted neural network layers (Linear, Activation, Conv2D, AvgPool2D), split learning protocols, and comprehensive benchmarking tools. CURE enables secure split learning while substantially improving communication and parallelization compared to prior work.

### Security/Privacy Issues and Ethical Concerns

This artifact does not contain any security vulnerabilities, malware, or personally identifiable information. The MNIST dataset used for evaluation is publicly available. There are no ethical concerns with making this artifact publicly available.

## Basic Requirements

### Hardware Requirements

1. **Minimum Requirements**: 8 GB RAM, 4-core CPU. Can run on a laptop (No special hardware requirements).
2. **Evaluation Hardware**: The experiments reported in the paper were performed on a machine with 2× Intel Xeon E5-2650 v3 @ 2.30GHz (20 physical cores, 40 threads with HT) and 251 GB RAM.

### Software Requirements

1. **OS**: Linux (Ubuntu 20.04+ recommended) or macOS 12+.
2. **OS Packages**: Git, Make.
3. **Container**: Docker 20.10+ (Optional).
4. **Compiler**: Go 1.23 or higher.
5. **Dependencies**: All Go dependencies (including Lattigo v6) are managed automatically via Go modules. Python 3 is required for result comparison scripts.
6. **ML Models**: The artifact includes definitions for MNIST-MLP, LeNet, and BCW models in `nn/bench/models.go`.
7. **Datasets**: MNIST. For timing benchmarks, synthetic data is generated automatically. For accuracy experiments, the artifact expects MNIST data in `data/mnist/raw/`.

### Estimated Time and Storage Consumption

- **Time**: ~15 minutes for quick validation; ~1-2 hours for full benchmarks.
- **Storage**: ~500 MB (including source, dependencies, and build artifacts).

## Environment

### Accessibility

The artifact is available on GitHub:
https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning/tree/v1.0-popets

### Set up the environment

#### Option 1: Native Installation

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
git clone https://github.com/CRYPTO-KU/CURE-Privacy-Preserving-Split-Learning.git
cd CURE-Privacy-Preserving-Split-Learning
git checkout v1.0-popets

# Build Docker image
docker build -t cure:latest .
```

### Testing the Environment

To verify the installation, run the unit tests.

Native:
```bash
make test
```

Docker:
```bash
docker run --rm -it cure:latest go test ./... -v -short
```

Expected output: All tests should pass with `PASS` status.

## Artifact Evaluation

### Main Results and Claims

| Claim | Paper Section | Experiment | Expected Result |
|-------|---------------|------------|-----------------|
| **C1**: HE layers are numerically correct | §5.1 | Exp 1 | RMS error < 1e-4 |
| **C2**: Split learning with HE is practical | §6, Tables 2-4 | Exp 2-3 | Timings within 5% |
| **C3**: Parallelization improves performance | §6.3 | Exp 3 | Near-linear speedup |

### Experiments

#### Experiment 1: Layer Correctness Tests (Claim C1)

- **Time**: ~5 minutes
- **Storage**: Minimal

Validate that HE-encrypted layers produce numerically correct results compared to plaintext.

```bash
go test ./nn/... -v -run Correctness
```

#### Experiment 2: Quick Benchmark (Claims C2, C3)

- **Time**: ~10-15 minutes
- **Storage**: ~10 MB

Reproduce benchmark results with reduced parameters for faster evaluation.

```bash
cd cmd/benchmarks
go build -o benchmark .
./benchmark --logn=13 --cores=1
```

#### Experiment 3: Full Benchmark Suite (Claims C2, C3)

- **Time**: ~1-2 hours
- **Storage**: ~50 MB

Reproduce the complete benchmark results from the paper.

```bash
cd cmd/benchmarks
# Run benchmarks with different core counts
for cores in 1 2 4 8; do
    ./benchmark --logn=13 --cores=$cores
done
```

## Limitations

1. **Hardware Variability**: Absolute benchmark times depend on CPU performance. Results should be compared relatively.
2. **Memory Requirements**: Large ring sizes (logN ≥ 15) require 16+ GB RAM.
3. **Dataset**: MNIST processed binary files are excluded. Synthetic data is used for timing benchmarks.

## Notes on Reusability

This artifact is designed for reuse and extension:
1. **Adding New Layers**: Implement the `nn.Module` interface in `nn/layers/`.
2. **Custom Models**: Define architectures in `nn/bench/models.go`.
3. **Integration**: Import as a Go module: `import "cure_lib/core/ckkswrapper"`.
