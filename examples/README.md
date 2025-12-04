# CURE_lib Examples

This directory contains examples demonstrating the CURE_lib privacy-preserving deep learning framework.

## Prerequisites

Build the binaries:
```bash
cd /path/to/CURE_lib
make build
```

This creates the following binaries in `bin/`:
- `cure-train` - Standalone trainer
- `cure-server` - Server-side split learning
- `cure-client` - Client-side split learning
- `cure-infer` - Encrypted inference
- `cure-bench` - Benchmarking tool

## Example 1: MNIST-128 Training (Synthetic Data)

Train a 784→128→32→10 model on synthetic MNIST-like data:

```bash
# Plaintext training (fast, for development)
./bin/cure-train --model=mnist --epochs=5 --samples=100 --encrypted=false --verbose

# Encrypted training (slow, but privacy-preserving)
./bin/cure-train --model=mnist --epochs=2 --samples=10 --encrypted=true --logN=13 --verbose
```

Expected output:
```
╔══════════════════════════════════════════════════════════════╗
║                    CURE_lib Trainer                          ║
║        Privacy-Preserving Deep Learning with HE              ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
  Model:         mnist
  Epochs:        5
  Learning Rate: 0.0100
  LogN:          13 (slots=4096)
  Encrypted:     true
  Samples:       10

Model architecture:
  [0] Linear(784 -> 128, encrypted=true)
  [1] Activation(encrypted=true)
  [2] Linear(128 -> 32, encrypted=true)
  [3] Activation(encrypted=true)
  [4] Linear(32 -> 10, encrypted=true)

Epoch 1/5 | Loss: 2.345678 | Time: 12.34s
Epoch 2/5 | Loss: 2.123456 | Time: 11.89s
...
Training complete!
```

## Example 2: MNIST-64 Training

Train a smaller 784→64→10 model:

```bash
./bin/cure-train --model=mnist --epochs=5 --samples=50 --encrypted=false
```

## Example 3: BCW-64 Training

Train on Breast Cancer Wisconsin-like synthetic data (64→32→16→10):

```bash
./bin/cure-train --model=bcw --epochs=10 --samples=100 --encrypted=false --verbose
```

## Example 4: Save and Load Weights

Train and save weights to a JSON file:

```bash
# Train and save
./bin/cure-train --model=mnist --epochs=5 --samples=100 --encrypted=false --output=mnist_weights.json

# Run inference with saved weights
./bin/cure-infer --weights=mnist_weights.json --logN=13 --encrypted=true
```

## Example 5: Split Learning (Server/Client)

This demonstrates privacy-preserving split learning where the server only sees encrypted data.

**Terminal 1 (Server):**
```bash
# Create named pipes for communication
mkfifo /tmp/to_server /tmp/from_server

# Start server
./bin/cure-server --logN=13 --model=mnist --verbose < /tmp/to_server > /tmp/from_server
```

**Terminal 2 (Client):**
```bash
# Start client
./bin/cure-client --logN=13 --model=mnist --epochs=3 --samples=10 --verbose > /tmp/to_server < /tmp/from_server
```

## Example 6: Encrypted Inference Demo

Run encrypted inference with random weights (demo mode):

```bash
./bin/cure-infer --logN=13 --encrypted=true --verbose

# Expected output:
# Top 3 predictions:
#   1. Class 5: 0.1523
#   2. Class 3: 0.1234
#   3. Class 7: 0.1012
```

## Example 7: Benchmarking

Run micro-benchmarks to measure HE operation performance:

```bash
# Quick benchmark
./bin/cure-bench --logN=13 --cores=4

# Full benchmark with CSV output
./bin/cure-bench --logN=13 --cores=8 --output=results.csv
```

## Parameter Reference

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--logN` | 13 | Ring dimension (13-16). Higher = more precision but slower |
| `--encrypted` | true | Use HE encryption |
| `--verbose` | true | Verbose output |
| `--lr` | 0.01 | Learning rate |
| `--epochs` | 5 | Training epochs |
| `--samples` | 100 | Number of synthetic samples |

### HE Parameter Guidelines

| LogN | Slots | Memory | Use Case |
|------|-------|--------|----------|
| 13 | 4096 | ~2GB | Development, small models |
| 14 | 8192 | ~4GB | Medium models |
| 15 | 16384 | ~8GB | Large models, production |
| 16 | 32768 | ~16GB | Maximum precision |

## Troubleshooting

### Out of Memory
Reduce `--logN` or use fewer samples:
```bash
./bin/cure-train --logN=13 --samples=10
```

### Slow Training
Use plaintext mode for development:
```bash
./bin/cure-train --encrypted=false
```

### Level Exhaustion Errors
The library uses cheat-strap (decrypt/re-encrypt) for level refresh. If you see level errors, the model may be too deep for the current parameter set.

## Next Steps

1. Replace synthetic data with real datasets
2. Implement real bootstrapping (remove cheat-strap dependency)
3. Add more model architectures (CNN, ResNet)
4. Optimize for multi-core parallelism
