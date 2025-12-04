# CURE_lib Artifact Documentation

This document describes the deliverables and artifacts of the CURE_lib project - a Go library for privacy-preserving deep learning using Homomorphic Encryption (HE) with CKKS via Lattigo v6.

## Project Structure

```
cure_lib/
├── core/                   # CKKS wrapper and HE context management
│   └── ckkswrapper/
│       ├── ckkswrapper.go  # HeContext, ServerKit, key generation
│       ├── ckkswrapper_test.go
│       └── debug.go        # Debug helpers (build tag: debug)
├── nn/                     # Neural network modules
│   ├── layers/             # Layer implementations
│   │   ├── linear.go       # Fully-connected layer (plain + HE)
│   │   ├── conv.go         # 2D convolution (plain + HE)
│   │   ├── conv1d.go       # 1D convolution
│   │   ├── activation.go   # Polynomial activations (ReLU3)
│   │   ├── avgpool2d.go    # Average pooling (plain + HE)
│   │   └── flatten.go      # Tensor flattening
│   ├── bench/              # Benchmarking utilities
│   │   ├── models.go       # Pre-built model architectures
│   │   ├── microbench.go   # Micro-benchmark helpers
│   │   └── aggregate.go    # Result aggregation
│   ├── module.go           # Module interface definition
│   ├── he_models.go        # Split HE model implementation
│   └── loss.go             # Loss functions (CrossEntropy)
├── tensor/                 # Basic tensor operations
│   └── tensor.go           # N-dimensional tensor type
├── utils/                  # Utilities
│   ├── timing.go           # Timing statistics (configurable output)
│   └── config.go           # Configuration helpers
├── cmd/                    # Command-line tools
│   ├── train/              # cure-train: Local training CLI
│   ├── server/             # cure-server: Split learning server
│   ├── client/             # cure-client: Split learning client
│   ├── infer/              # cure-infer: Encrypted inference
│   ├── benchmarks/         # cure-bench: Benchmarking CLI
│   └── bench_rerun/        # Benchmark re-run tool
├── split/                  # Split learning protocol
│   └── protocol.go         # Gob-based RPC messages
├── data/                   # Dataset placeholders
│   └── mnist/              # MNIST data directory
├── go.mod                  # Go module definition
└── README.md               # Project documentation
```

## Core Components

### 1. CKKS Wrapper (`core/ckkswrapper/`)

The CKKS wrapper provides a simplified interface to Lattigo v6's CKKS scheme:

- **`HeContext`**: Client-side context containing:
  - CKKS parameters (configurable logN, default 15)
  - Secret key, encoder, encryptor, decryptor
  
- **`ServerKit`**: Server-side context containing:
  - Public parameters, encoder, evaluator
  - Pre-generated rotation and relinearization keys

```go
heCtx := ckkswrapper.NewHeContext()              // Default logN=15
heCtx := ckkswrapper.NewHeContextWithLogN(13)    // Custom ring size
serverKit := heCtx.GenServerKit([]int{1,2,4,-1}) // Required rotations
```

### 2. Neural Network Layers (`nn/layers/`)

All layers implement the `nn.Module` interface:
```go
type Module interface {
    Forward(input interface{}) (interface{}, error)
    Backward(gradOut interface{}) (interface{}, error)
    Encrypted() bool
    Levels() int
}
```

#### Linear Layer
- Plain and HE forward/backward paths
- Rotation-based matrix-vector multiplication
- Slot masking for correctness

#### Conv2D Layer
- Channel-block packing for efficient HE operations
- BSGS (Baby-Step Giant-Step) style optimizations
- Configurable kernel sizes

#### Activation Layer
- Polynomial approximation of ReLU (ReLU3)
- Horner's method for efficient HE evaluation
- Degree-3 polynomial: 2 levels consumed

#### AvgPool2D Layer
- Slot-based average pooling
- Rotation-accumulate pattern
- 1 level consumed

### 3. Split Learning Model (`nn/he_models.go`)

Demonstrates split learning with HE:
- Server-side: Linear + Activation (encrypted)
- Client-side: Linear layers (plaintext)
- Gradient encryption/decryption at split point

### 4. Timing & Instrumentation (`utils/`)

Configurable timing output:
```go
utils.Verbose = false  // Suppress timing output
utils.Output = file    // Redirect to file
utils.PrintTimingStats(stats, steps)
```

Operation counting via `WrappedEvaluator`:
- Rotation, multiplication, relinearization counts
- Per-phase statistics

## Build Instructions

### Requirements
- Go 1.23+
- Lattigo v6 (via Go modules)
- Docker (optional, for containerized builds)

### Using Makefile
```bash
make build          # Build all binaries to bin/
make test           # Run tests with race detection
make release        # Cross-compile Linux binaries
make docker         # Build Docker image
make clean          # Remove build artifacts
```

### Manual Build
```bash
go mod download
go build -o bin/cure-train ./cmd/train
go build -o bin/cure-server ./cmd/server
go build -o bin/cure-client ./cmd/client
go build -o bin/cure-infer ./cmd/infer
go build -o bin/cure-bench ./cmd/benchmarks
```

### Docker
```bash
docker build -t cure:latest .
docker run -it cure:latest cure-infer --help
```

## CLI Tools

### cure-train
Local training on MNIST/BCW datasets:
```bash
./bin/cure-train --model mnist --epochs 10 --samples 100 --lr 0.01
```

### cure-server
Split learning server (waits for client connections):
```bash
./bin/cure-server --port 8080 --logN 13 --model mnist
```

### cure-client
Split learning client:
```bash
./bin/cure-client --addr localhost:8080 --epochs 5 --samples 100
```

### cure-infer
Encrypted inference:
```bash
./bin/cure-infer --model lenet --weights model.bin --input image.raw
```

Options:
- `--plain`: Run plaintext inference (for comparison)
- `--logN`: Ring dimension (default 13)
- `--verbose`: Enable timing output

### Testing
```bash
go test ./...
```

### Running Benchmarks
```bash
cd cmd/benchmarks
go build -o benchmark
./benchmark --logN=13 --cores=4
```

## API Usage Examples

### Basic HE Forward Pass
```go
heCtx := ckkswrapper.NewHeContext()
lin := layers.NewLinear(128, 64, true, heCtx)
lin.SyncHE()

act, _ := layers.NewActivation("ReLU3", true, heCtx)

ctIn := encryptInput(inputData, heCtx)
ctHidden, _ := lin.ForwardCipherMasked(ctIn)
ctOut, _ := act.ForwardCipher(ctHidden)
```

### Split Learning Training Step
```go
model := nn.NewSplitHEModel([]int{784, 128, 64, 10}, heCtx, 0.01, stats)
loss, _ := model.TrainStep(heCtx, inputTensor, labelTensor)
```

## Configuration Options

### CKKS Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| logN | 15 | Ring dimension (2^logN slots) |
| LogQ | 6×45 bits | Modulus chain |
| Scale | 2^45 | Default scaling factor |

### Verbose Output
Set `utils.Verbose = false` to disable all timing and statistics output.

## Benchmarking Results

Pre-built models available in `nn/bench/models.go`:
- MNIST FC: 784-128-32-10
- LeNet: Conv-Pool-Conv-Pool-FC
- BCW FC: 64-32-16-10
- Audio1D: 1D CNN for audio
- ResNet Block: Conv layers

## Limitations

1. **Performance**: CPU-intensive operations; optimize for small dimensions during development
2. **Level Budget**: Monitor level consumption; use cheat-strap for extended computation
3. **Plaintext Precision**: CKKS introduces small errors; validate against plaintext reference
4. **Memory**: Large ring sizes require significant memory

## License

Research-oriented codebase for experimentation and benchmarking.

## References

- Lattigo v6: https://github.com/tuneinsight/lattigo
- CKKS Scheme: Cheon, Kim, Kim, Song (2017)
