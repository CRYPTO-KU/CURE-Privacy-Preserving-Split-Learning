### ✅ Design choices locked-in

| Dimension               | Decision                                                                                                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Language**            | Pure **Go**                                                                                                                                                           |
| **Platforms**           | Build scripts stay generic; no code changes needed.                                                                                                                   |
| **Default CKKS params** | Ring-size = $2^{13}$ (8192 slots) chain = 6×45-bit.  <br>*(All accountant formulas stay valid; we can add bigger presets later.)*                                     |
| **Bootstrapping**       | **“Cheat-strap”** only: decrypt → reencrypt under same key when accountant says *bootstrap here*.  (Add code comment so swapping to real `ctx.Bootstrap` is trivial.) |
| **External deps**       | Keep **minimal**: `lattigo/v6`, std-lib, and `golang.org/x/sync/errgroup` which we already vendored.  No gRPC for v0.1.                                               |

These clarifications slot straight into the PRD from my last message.
Below is the updated action plan plus concrete Git scaffolding steps so an AI-agent (or human) can start **Day 0**.

---

## 1  Repo bootstrap script

```bash
# run once in an empty folder
go mod init cure
go get github.com/tuneinsight/lattigo/v6@v6.0.0
go get golang.org/x/sync@v0.15.0   # errgroup only
mkdir -p core tensor nn schedule engine split cmd examples
touch core/ckkswrapper.go nn/module.go README.md
git add .
git commit -m "Scaffold CURE-NN repo – ckkswrapper & module stubs"
```

---

## 2  Immediate backlog (create as GitHub Issues)

| Prio | Title                                             | Summary / Acceptance                                                                                                                                                     |                                                                                 |                                                                  |
| ---- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| P0   | **ckkswrapper: context factory + Encryptor pool** | Wrap Lattigo params (13-bit ring, 6×45 primes) + helpers `NewCtx()`, `GetWorkerCtx()` (one per goroutine). Unit test: plain → enc → dec round-trip.                      |                                                                                 |                                                                  |
| P0   | **Tensor package skeleton**                       | Basic ND-array over `[]float64`, ops `MatMul`, `Add`, `ReluPlain`. 100 % Go/CPU, no HE.                                                                                  |                                                                                 |                                                                  |
| P1   | **Module interface & Sequential**                 | `type Module interface { Forward(x Tensor) (Tensor,error); Backward(grad Tensor) (Tensor,error); Levels() int; Encrypted() bool }`. Provide `nn.Sequential{}` container. |                                                                                 |                                                                  |
| P1   | **Linear layer – plain + HE**                     | Plain: `Wx+b`. HE: rotation-dot implementation using EnsureBudget cheat. Unit tests compare first slot to plain within 1e-3.                                             |                                                                                 |                                                                  |
| P1   | **Activation poly (degree 3)**                    | Hard-code Chebyshev coeffs; plain vs. HE eval.                                                                                                                           |                                                                                 |                                                                  |
| P2   | **Accountant pass**                               | Walk graph, sum `Levels()`, insert `schedule.BootstrapOp` nodes where necessary. Use cheat-strap.                                                                        |                                                                                 |                                                                  |
| P2   | **Worker-pool forward**                           | `engine.ForwardPool` similar to prototype; consumes slice of samples; each worker gets its own ctx from pool.                                                            |                                                                                 |                                                                  |
| P2   | **Back-prop for Linear (plain + HE gradients)**   | Match signature in Module. HE side just returns encrypted grad of weight matrix (rotations + InnerSum).                                                                  |                                                                                 |                                                                  |
| P3   | **Split RPC placeholder**                         | For v0.1 just use file-pipe serialization (`gob`) of ciphertext slices to keep deps minimal.                                                                             |                                                                                 |                                                                  |
| P3   | **CLI skeleton**                                  | `cmd/train` and \`cmd/server                                                                                                                                             | client\` accept JSON model spec and data path; run one epoch on synthetic data. |                                                                  |
| P4   | **Examples**                                      | `examples/mnist_lenet.go` builds model \`784-128-                                                                                                                        | cut                                                                             | -32-10\`, runs synthetic train, prints accountant boot schedule. |

*(Add labels `core`, `nn`, `parallel`, `split`, `cli`.)*

---

## 3  Coding guidelines

* **No shared `rlwe.Evaluator` across goroutines.** Grab from context pool.
* **Levels bookkeeping**: every Module constructor must set a constant `levelsUsed int` (e.g., Linear-HE = 1, Activation-d3 = 2). The accountant relies on this being correct.
* **Cheat-strap implementation**

```go
func CheatBootstrap(ct *ckks.Ciphertext, ctx *ckks.Context) *ckks.Ciphertext {
    pt := ctx.Decoder.Decode(ctx.SecretKey, ct)
    return ctx.EncryptorPK.EncryptNew(pt)
}
```

* **Synthetic datasets**: helper `data.Random(batch, dim)` returns `Tensor` with `rand.NormFloat64()` entries and labels `rand.Intn(numClass)`; we will not bundle MNIST/BCW files.

---

## 4  Deliverables for v0.1 tag

1. `make test` passes all unit tests (Forward/Backward correctness, accountant schedule).
2. `make release` emits **four static binaries**:

   * `cure-train` – standalone single-process trainer.
   * `cure-server` / `cure-client` – split learning demo using gob over stdin/stdout.
   * `cure-infer` – loads saved weights, runs encrypted inference.
3. `examples/README.md` shows three commands reproducing the MNIST-128, MNIST-64, BCW-64 splits on synthetic data, printing loss & accountant log.

---

## 5  Anything still unclear?

* **Optimizer scope** – only vanilla SGD, or need momentum & Adam in v0.1?
* **Saving weights** – JSON (plain + base64 ciphertext) OK?
* **Desired concurrency default** – use `runtime.NumCPU()` workers or fixed 8?

Let me know, and I’ll transform the issue list into code scaffolding.
