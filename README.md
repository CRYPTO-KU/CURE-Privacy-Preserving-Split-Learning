## CURE: Privacy-Preserving Split Learning Done Right

A research-oriented Go library for privacy-preserving deep learning with Homomorphic Encryption (HE) using CKKS via Lattigo v6. This library provides the official implementation of the protocols and optimizations presented in our paper, "CURE: Privacy-Preserving Split Learning Done Right," published in PoPETs 2026. It provides:

- Encrypted-ready neural network layers (Linear, Activation, Conv2D, AvgPool2D, utilities)
- CKKS setup helpers and client/server key split utilities
- Split learning example that performs server-side HE forward and client-side plaintext training
- Timing utilities and micro-bench support to study performance and HE operation counts

This codebase is intended for experimentation and benchmarking rather than production use.


### Features
- **CKKS wrapper**: Simple construction of `HeContext` (client-owned) and `ServerKit` (server-owned) with rotation/relin keys.
- **HE layers**: `Linear`, `Activation` (polynomial, e.g. ReLU3), `Conv2D`, `AvgPool2D` with HE forward/backward/update flows.
- **Split HE model**: Example `SplitHEModel` that encrypts input, runs server-side HE layers, decrypts, and continues training in plaintext.
- **Instrumentation**: `utils.TimingStats` and wrapped evaluators to count HE ops and timings.
- **Batched and packed operations** where appropriate, with rotation key planning.


### Requirements
- Go 1.23+
- Lattigo v6 (brought in via Go modules)
- macOS/Linux recommended; tests rely on CPU bigints and can be compute intensive

Install Go dependencies:

```bash
cd path/to/CURE_lib
go mod download
```


### Project structure
- `core/ckkswrapper`: CKKS helpers
  - `HeContext`: client-side params, secret key, encoder, encryptor, decryptor
  - `ServerKit`: server-side params, encoder, evaluator, public/eval keys
- `nn/layers`: Neural network layers with plaintext and HE paths
  - `linear.go`: HE-masked linear forward/backward/update helpers
  - `activation.go`: Polynomial activations (e.g., ReLU3) with HE evaluation via Horner’s method
  - `conv.go`: 2D convolution with channel-block packing, BSGS-style optimization hooks
  - `avgpool2d.go`: Average pooling with slot masking and rotations
  - `wrapped_evaluator.go`: Operation counting wrapper for evaluators
  - `onelevel.go`: One-level packed/slotwise experiments
- `nn/he_models.go`: `SplitHEModel` showing end-to-end split learning with HE server path
- `tensor`: Minimal tensor type and basic ops (MatMul, Add, ReLU)
- `utils`: Timing and config helpers
- `data/mnist`: Placeholder for datasets (user-supplied)


### Quick start
Below are minimal code sketches illustrating the main flows. These are snippets; see files for full details.

1) Initialize CKKS client context and produce a server kit

```go
import (
    "cure_lib/core/ckkswrapper"
)

heCtx := ckkswrapper.NewHeContext()              // defaults to logN=15
serverKit := heCtx.GenServerKit([]int{1,2,4,-1}) // rotations you need
_ = serverKit // hand to server side
```

2) Build an HE Linear + Activation server block and run encrypted forward

```go
import (
    "cure_lib/nn/layers"
    "github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

inDim, outDim := 128, 64
lin := layers.NewLinear(inDim, outDim, true, heCtx)
// initialize weights/bias, then
lin.SyncHE()
act, _ := layers.NewActivation("ReLU3", true, heCtx)

// Encrypt input vector (length inDim)
pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
heCtx.Encoder.Encode(make([]complex128, heCtx.Params.MaxSlots()), pt) // fill with your data
ctIn, _ := heCtx.Encryptor.EncryptNew(pt)

ctHidden, _ := lin.ForwardCipherMasked(ctIn)
ctOut, _ := act.ForwardCipher(ctHidden)
_ = ctOut
```

3) Split learning example (server HE, client plaintext)

```go
import (
    "cure_lib/nn"
    "cure_lib/core/ckkswrapper"
    "cure_lib/tensor"
    "cure_lib/utils"
)

heCtx := ckkswrapper.NewHeContext()
stats := &utils.TimingStats{}
arch := []int{784, 128, 64, 10} // input, server hidden, client hidden, output
model := nn.NewSplitHEModel(arch, heCtx, 0.01, stats)

// One training step: input and one-hot label as tensors
x := tensor.New(arch[0])
y := tensor.New(10); y.Data[3] = 1 // class 3
loss, err := model.TrainStep(heCtx, x, y)
_ = loss; _ = err
```


### Key concepts and APIs
- `ckkswrapper.NewHeContext()` builds a client context with default CKKS params (`logN=15`, `LogQ/LogP` set in `core/ckkswrapper/ckkswrapper.go`).
- `HeContext.GenServerKit(rotations)` pre-creates relinearization and rotation keys; select rotations based on your layer needs.
- Layers created with `encrypted=true` use HE pathways; otherwise plaintext.
- HE layers often require a preparation step:
  - `Linear.SyncHE()` to encode and encrypt weights and bias.
  - `Conv2D.SetDimensions(h, w)` then `Conv2D.SyncHE()` to compute masks and packing.
  - `AvgPool2D.SetDimensions(h, w)` as needed.
- Backward/Update:
  - Plaintext layers implement `Backward` and `Update` for SGD.
  - Encrypted layers implement `BackwardHE`/`UpdateHE`-style flows; some updates operate on cached ciphertexts or plaintext shadows depending on the file’s state.


### Parameter selection and rotations
HE performance depends on parameter sizes, level consumption, and rotation key coverage.
- `Activation(ReLU3)` consumes ~2 levels; `Conv2D` typically ~2; `AvgPool2D` ~1.
- Generate rotation keys only for indices actually used by your model. See `NewLinear` and `Conv2D.SetDimensions` for exact rotations.


### Benchmarks and tests
The repository includes unit tests and micro-bench helpers under `nn/layers/*_test.go` and `nn/bench`.


Micro-bench examples (see files in `nn/bench/`):
- `bench/microbench.go`, `bench/models.go`, and references in `layers/*_bench_test.go`.

Timing and operation counts:
- Use `utils.TimingStats` to record durations across steps.
- Use wrappers in `wrapped_evaluator.go` or add counters in your code to track `Rotate/Mul/Relin/Rescale/Add`.


### Data
- `data/mnist` is a placeholder. Provide your own dataset loader when integrating end-to-end examples.


### Notes & limitations
- This library is experimental; APIs may change and some HE updates use simplified flows.
- Many operations are CPU- and memory-intensive; prefer small dimensions when experimenting.
- Always validate level/scale alignment when combining ciphertexts and plaintexts.

### Citation

Please cite this work as

@article{Kanpak2026CURE,
  author    = {Kanpak, Halil Ibrahim and Shabbir, Aqsa and Gen{\c{c}}, Esra and K{\"{u}}p{\c{c}}{\"{u}}, Alptekin and Sav, Sinem},
  title     = {CURE: Privacy-Preserving Split Learning Done Right},
  journal   = {Proceedings on Privacy Enhancing Technologies},
  volume    = {2026},
  number    = {2},
  year      = {2026},
  publisher = {Sciendo}
}

### Acknowledgments
This work is supported by TÜBİTAK (the Scientific and Technological Research Council of Türkiye) projects 123E462 and 124N941. The authors used ChatGPT-4o~\cite{openai2024gpt4o} to enhance the manuscript by shortening sentences, correcting typos, and improving grammar. We acknowledge contributions of Ercüment Çiçek and Yaman Yağız Taşbağ.
