# CURE_lib Repository Documentation

## Project Overview

**CURE_lib** is a Go-based homomorphic encryption library for neural networks that enables privacy-preserving machine learning computations. It uses the CKKS scheme from the Lattigo v6 library and supports both plaintext and encrypted neural network operations.

### Key Features
- **Dual-Mode Operations**: All layers support both plaintext and homomorphic encryption (HE) modes
- **Split Learning**: Client-server architecture where sensitive computations happen on encrypted data
- **SIMD Optimization**: Leverages CKKS SIMD capabilities for batch processing
- **Modular Design**: Layer-based neural network construction with mix-and-match capabilities
- **Benchmarking Suite**: Comprehensive performance measurement tools

### Dependencies
- **Go Version**: 1.23.0
- **Primary Dependency**: `github.com/tuneinsight/lattigo/v6 v6.1.1` (CKKS homomorphic encryption)
- **Testing**: `github.com/stretchr/testify v1.8.0`

---

## Directory Structure

### `/core/` - Core Cryptographic Components
Contains fundamental cryptographic abstractions and CKKS parameter management.

#### `/core/ckkswrapper/`
**Purpose**: Wraps Lattigo's CKKS implementation with simplified interfaces

**Key Files:**
- `ckkswrapper.go` - Main CKKS wrapper implementation
- `debug.go` - Debug utilities for comparing plaintext vs encrypted results
- `ckkswrapper_test.go` - Basic encryption/decryption tests

**Core Structures:**
```go
// HeContext - Complete cryptographic context (CLIENT-SIDE)
type HeContext struct {
    Params    ckks.Parameters
    Sk        *rlwe.SecretKey    // SECRET KEY - only on client
    Encoder   *ckks.Encoder
    Encryptor *rlwe.Encryptor
    Decryptor *rlwe.Decryptor
}

// ServerKit - Public components for server (NO SECRET KEY)
type ServerKit struct {
    Params    ckks.Parameters
    Encoder   *ckks.Encoder
    Evaluator *ckks.Evaluator
    Pk        *rlwe.PublicKey
    Evk       rlwe.EvaluationKeySet  // Rotation/relinearization keys
}
```

**Key Functions:**
- `NewHeContext()` - Creates HE context with default logN=15
- `NewHeContextWithLogN(logN int)` - Creates context with custom ring size
- `GenServerKit(rotationKeys []int)` - Generates server evaluation keys
- `Refresh(ct *rlwe.Ciphertext)` - "Bootstrap" by decrypt/re-encrypt (resets noise)

**Important Implementation Details:**
- Default CKKS parameters: `LogQ: []int{60, 40, 40, 40, 38}`, `LogP: []int{60}`
- Default scale: 40 bits
- Ring dimension configurable via logN (default 15 = 32768 slots)

### `/nn/` - Neural Network Implementation
Contains the core neural network framework with support for both plaintext and HE operations.

#### `/nn/module.go` - Base Module Interface
**Core Interface:**
```go
type Module interface {
    Forward(input interface{}) (interface{}, error)
    Backward(gradOut interface{}) (interface{}, error)
    Encrypted() bool
    Levels() int  // HE levels consumed
}
```

**Sequential Model:**
```go
type Sequential struct {
    Layers []Module
}
```

#### `/nn/layers/` - Layer Implementations

##### `/nn/layers/linear.go` - Fully Connected Layer
**Most Complex Layer** - Supports matrix-vector products in HE domain

**Key Structure:**
```go
type Linear struct {
    W, B *tensor.Tensor           // Plaintext weights/bias
    heCtx *ckkswrapper.HeContext
    serverKit *ckkswrapper.ServerKit
    weightCTs []*rlwe.Ciphertext  // Row-wise encrypted weights
    biasPT *rlwe.Plaintext        // Plaintext bias
    maskPT *rlwe.Plaintext        // Slot-0 extraction mask
    encrypted bool
    // Shadow plaintext for debugging
    wShadow, bShadow, lastInputShadow *tensor.Tensor
}
```

**Critical Implementation Details:**
- **Rotation Keys Required**: The linear layer requires specific rotation patterns for CKKS operations
- **Forward Pass Rotations**:
  - Powers of 2 for tree-sum (input dimension): `1, 2, 4, 8, ...`
  - Negative rotations for slot assembly: `-0, -1, -2, ..., -(outDim-1)`
- **Backward Pass Rotations**:
  - Powers of 2 for gradient tree-sum (output dimension)
  - **NEGATIVE powers of 2 for broadcast**: `-1, -2, -4, -8, ...` (input dimension)
  - Positive rotations for slot isolation: `+0, +1, +2, ..., +(outDim-1)`

**⚠️ CRITICAL LATTIGO IMPLEMENTATION NOTE:**
The negative rotations (`-j`) are essential for the matrix-vector product implementation. They are used to:
1. **Slot Assembly**: After computing dot products, rotate results to correct output positions
2. **Broadcast Operations**: Distribute scalar values across SIMD slots for gradient computation
3. **Slot Isolation**: Extract individual gradient components for weight updates

**Key Functions:**
- `NewLinear(inDim, outDim, encrypted, heCtx)` - Constructor
- `SyncHE()` - Encodes plaintext weights into HE format
- `ForwardCipherMasked(ct)` - HE matrix-vector multiplication
- `BackwardWithDebug(gradOut, t)` - HE gradient computation
- `UpdateHE(lr)` - HE weight updates

##### `/nn/layers/activation.go` - Activation Functions
**Purpose**: Polynomial approximations of non-linear functions

**Supported Polynomials:**
```go
var SupportedPolynomials = map[string]Poly{
    "ReLU3": {
        Coeffs: []float64{0.3183099, 0.5, 0.2122066, 0},
        Degree: 3,
        Levels: 2,  // Consumes 2 HE levels
    },
    "ReLU3_deriv": {
        Coeffs: []float64{0.5, 0.4244},
        Degree: 1, 
        Levels: 1,
    },
}
```

##### `/nn/layers/conv.go` - 2D Convolutional Layer
**Complex HE Implementation** with channel-block packing and BSGS optimization

**Key Features:**
- **Channel-block packing**: Groups output channels into SIMD slots
- **Tree-sum fusion**: Optimizes spatial convolutions
- **Rotation patterns**: `-(dy*inW + dx)` for forward, `+(dy*inW + dx)` for backward

##### `/nn/layers/` - Other Layers
- `flatten.go` - Reshaping layer (no-op for HE)
- `avgpool2d.go` - Average pooling with masking-based implementation
- `conv1d.go` - 1D convolution (wrapper around Conv2D)
- `pool1d.go` - 1D max pooling
- `residual.go` - Residual blocks for ResNet-style architectures
- `onelevel.go` - Optimized single-level HE operations

#### `/nn/bench/` - Benchmarking Suite
**Purpose**: Performance measurement and model architecture definitions

**Key Files:**
- `models.go` - Pre-defined network architectures (MNIST-FC, LeNet, ResNet, etc.)
- `microbench.go` - Individual layer timing
- `aggregate.go` - Full model benchmarking with cut analysis

**Pre-defined Models:**
- `BuildMNISTFC()` - 784→128→32→10 fully connected
- `BuildLeNet()` - Classic CNN architecture
- `BuildBCWFC()` - 64→32→16→10 benchmark
- `BuildAudio1D()` - 1D CNN for audio processing
- `BuildResNetBlock()` - ResNet building blocks

#### `/nn/training.go` & `/nn/he_models.go` - Training Framework
**Split HE Training Implementation:**
```go
type SplitHEModel struct {
    // Server-side (HE) layers
    ServerLinear     *layers.Linear
    ServerActivation *layers.Activation
    
    // Client-side (plaintext) layers  
    ClientLinear1, ClientLinear2 *layers.Linear
    ClientActivation             *layers.Activation
    
    LossFn       *CrossEntropyLoss
    LearningRate float64
    Stats        *utils.TimingStats
}
```

### `/tensor/` - Tensor Operations
**Simple tensor implementation** for plaintext computations

**Key Structure:**
```go
type Tensor struct {
    Data  []float64  // Flat array storage
    Shape []int      // Dimension sizes
}
```

**Core Operations:**
- `New(shape...)` - Constructor
- `Add(a, b)` - Element-wise addition
- `MatMul(a, b)` - Matrix multiplication (2D only)
- `ReluPlain(a)` - Plaintext ReLU
- `At(indices...)` - Element access
- `Set(value, indices...)` - Element assignment

### `/dataset/` - Data Loading and Preprocessing
**MNIST Dataset Support** with caching

#### `/dataset/mnist/mnist.go`
**Advanced MNIST loader** with multiple format support:
- IDX format (raw and gzipped)
- TAR.GZ archives  
- Binary caching for fast reloading
- Automatic normalization to [0,1]

**Key Functions:**
- `Load(root, train)` - Main loading function with caching
- `PreprocessMNIST(images, labels)` - Flattening and one-hot encoding
- `GetSample(flatImages, oneHotLabels, idx)` - Single sample extraction

### `/utils/` - Utility Functions
**Configuration and timing utilities**

**Key Files:**
- `config.go` - Training configuration parsing and validation
- `timing.go` - Comprehensive performance measurement

**Timing Statistics:**
```go
type TimingStats struct {
    TotalTime, DataLoadingTime, HEInitTime time.Duration
    ModelInitTime, ForwardPassTime, BackwardPassTime time.Duration
    UpdateTime, EncryptionTime, DecryptionTime time.Duration
    ServerLinearTime, ServerActivationTime time.Duration
    ClientLinearTime, ClientActivationTime time.Duration
    LossComputationTime time.Duration
}
```

---

## Important Implementation Patterns

### CKKS/Lattigo Best Practices

#### 1. Rotation Key Management
**Critical for Performance**: Always generate rotation keys upfront
```go
// Linear layer rotation requirements
rots := []int{}
// Forward pass: powers of 2 for tree-sum
for step := 1; step < inDim; step *= 2 {
    rots = append(rots, step)
}
// Forward pass: negative rotations for slot assembly  
for j := 0; j < outDim; j++ {
    rots = append(rots, -j)  // NEGATIVE rotation crucial!
}
// Backward pass: negative powers of 2 for broadcast
for step := 1; step < inDim; step *= 2 {
    rots = append(rots, -step)  // NEGATIVE broadcast
}
serverKit = heCtx.GenServerKit(rots)
```

#### 2. Level Management
**CKKS operations consume modulus levels:**
- Multiplication: consumes 1 level (requires rescaling)
- Addition: no level consumption
- ReLU3 polynomial: consumes 2 levels
- **Always check `ct.Level()` before operations**

#### 3. Scale Management
**Maintain consistent scales:**
```go
// After multiplication, rescale to restore default scale
if err := evaluator.Rescale(ct, ct); err != nil {
    return err
}
// For level alignment
for ct1.Level() > ct2.Level() {
    evaluator.Rescale(ct1, ct1)
}
ct1.Scale = ct2.Scale  // Align scales
```

#### 4. Shadow Plaintext Debugging
**Always maintain shadow computations** for debugging:
```go
type Linear struct {
    // HE components
    weightCTs []*rlwe.Ciphertext
    // Shadow plaintext for debugging
    wShadow, bShadow, lastInputShadow *tensor.Tensor
}
```

### Common Mistakes to Avoid

#### 1. ❌ Missing Rotation Keys
```go
// WRONG: Missing required rotations
serverKit = heCtx.GenServerKit([]int{})  // Empty rotation keys

// CORRECT: Include all required rotations
rots := computeRequiredRotations(inDim, outDim)
serverKit = heCtx.GenServerKit(rots)
```

#### 2. ❌ Incorrect Rotation Directions
```go
// WRONG: Using positive rotations for slot assembly
rot, _ := evaluator.RotateNew(ct, +j)  // Should be negative!

// CORRECT: Use negative rotations for proper slot positioning
rot, _ := evaluator.RotateNew(ct, -j)
```

#### 3. ❌ Level Exhaustion
```go
// WRONG: Not checking levels before operations
result, _ := evaluator.MulNew(ct1, ct2)  // May fail if ct1.Level() == 0

// CORRECT: Check and refresh if needed
if ct1.Level() == 0 {
    ct1 = heCtx.Refresh(ct1)  // Reset to max level
}
result, _ := evaluator.MulNew(ct1, ct2)
```

#### 4. ❌ Scale Misalignment
```go
// WRONG: Operations with misaligned scales
result, _ := evaluator.AddNew(ct1, ct2)  // May fail if scales differ

// CORRECT: Align scales before operations
ct1.Scale = ct2.Scale
result, _ := evaluator.AddNew(ct1, ct2)
```

### Architecture Guidelines

#### Layer Design Patterns
1. **Dual Implementation**: Every layer supports both `encrypted bool` and plaintext modes
2. **Interface Compliance**: All layers implement `nn.Module` interface
3. **Shadow Debugging**: Maintain parallel plaintext computation paths
4. **Resource Management**: Cache inputs for backward pass, clean up temporary variables

#### Memory Management
- **Ciphertext Copying**: Use `ct.CopyNew()` for safe copies
- **Plaintext Reuse**: Cache encoded plaintexts when possible
- **Rotation Key Sharing**: Use `ServerKit.GetWorkerEvaluator()` for thread safety

---

## Testing and Validation

### Debug Mode
When compiled with `debug` build tag:
```go
//go:build debug
// Enables DebugCompare function in ckkswrapper/debug.go
```

### Test Patterns
- **Correctness Tests**: Compare HE vs plaintext results within tolerance
- **End-to-End Tests**: Full forward/backward/update cycles
- **Benchmark Tests**: Performance measurement across different parameters

### Common Test Utilities
```go
// Compare HE result with shadow plaintext
heCtx.DebugCompare(ct, shadowTensor, "layer_output", 1e-6, t)

// Time layer operations
fwd, bwd, upd, _, _ := TimeLayer(layer, slots, numRuns)
```

---

## Performance Optimization

### SIMD Exploitation
- **Slot Utilization**: Pack multiple values per ciphertext
- **Channel Blocking**: Group operations for better cache locality
- **Tree Reductions**: Use log(n) rotations instead of linear operations

### Memory Optimization  
- **Plaintext Caching**: Pre-encode frequently used constants
- **Rotation Key Precomputation**: Generate all required keys upfront
- **Worker Evaluators**: Use separate evaluators for parallel operations

### Level Conservation
- **Polynomial Degree Reduction**: Use lower-degree approximations when possible
- **Operation Reordering**: Minimize depth of multiplication chains
- **Strategic Refreshing**: Bootstrap only when absolutely necessary

---

## Recent Developments (PR History)

### PR #2: Multi-Layer Split Learning
- **Configurable architectures** via `ModelConfig`
- **Multi-layer HE server** with cached intermediate activations
- **SIMD optimizations** with image-per-ciphertext packing

### PR #3: Enhanced Configurability  
- **Flexible MLP architectures** with split point configuration
- **Multiple training modes**: Standard Split-HE, Fully HE, SIMD-Packed
- **Improved project structure** with dedicated `split/` package

---

## Future Development Notes

### Planned Enhancements
- **Additional activation functions** (Sigmoid, Tanh approximations)
- **Advanced CNN architectures** (BatchNorm, Dropout)
- **Optimization algorithms** (Adam, RMSprop with HE support)
- **Cross-platform deployment** tools

### Research Directions
- **Level optimization algorithms**
- **Adaptive polynomial approximations**
- **Distributed HE computation**
- **Hardware acceleration integration**

---

## Quick Reference

### Essential Commands
```bash
# Build and test
go build ./...
go test ./...

# Run benchmarks
go test -bench=. ./nn/bench/

# Debug mode
go build -tags debug ./...
```

### Key Environment Variables
- `GOMAXPROCS`: Controls parallelism
- Debug builds enable additional validation

### Critical Configuration Parameters
- **logN**: Ring dimension (13-16 typical, default 15)
- **LogQ**: Modulus chain (affects precision and levels)
- **DefaultScale**: Fixed-point scale (40 bits default)
- **Rotation Keys**: Must include all required rotations upfront

This documentation serves as a comprehensive reference for understanding, extending, and maintaining the CURE_lib codebase. Always refer to the latest code for implementation details, as this documentation reflects the current state of the repository.
