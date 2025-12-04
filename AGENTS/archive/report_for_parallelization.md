## Parallelization in Lattigo: Concise Reference

### 1. Overview

* **Goal**: Efficiently compute $k$ independent CKKS inner products in parallel over $N$ threads.
* **Pattern**: Tree‑sum (element‑wise multiply → relinearize → rescale → ⱼlog₂(vectorDim) rotations + adds).

### 2. Core Implementation Components

**ComputeInnerProduct**

```go
// 1) Mul → 2) Relinearize → 3) Rescale → 4) Tree-sum rotations
func ComputeInnerProduct(ct1, ct2, evaluator, dim) {...}
```

* Exactly one Mul, one Relin, one Rescale, and \~⌈log₂(dim)⌉ Rotate/Add steps.

**WorkerPool**

* `tasks chan InnerProductTask` distributes tasks.
* `workers []*OperationCounter` each holds its own `ckks.Evaluator` to avoid contention.
* Channel fan‑out/fan‑in with `sync.WaitGroup` for coordination.

**OperationCounter**

* Wraps `ckks.Evaluator` to track counts of rotations, muls, relins, rescales.
* `mu`‑protected counters updated on each call for profiling.

### 3. Scalability & Bottlenecks

| Threads | Speedup | Efficiency | Load Imbalance |    Primary Bottleneck    |
| :-----: | :-----: | :--------: | :------------: | :----------------------: |
|    1    |    1×   |    100%    |       0%       |      Relinearization     |
|    4    |   \~4×  |    \~90%   |       0%       |      Relinearization     |
|    8    |  \~4.6× |    \~58%   |      \~40%     | Relinearization (+imbl.) |

* **Relinearization** dominates 40–50% of compute time; grows with threads due to memory traffic.
* **Load imbalance** appears at high core counts with static partitioning.

### 4. Implementation Explanations for Agents

1. **Dedicated Evaluators**

   * Always call `ckks.NewEvaluator(params, evk)` per goroutine to avoid data races.
2. **Channel‐Based Task Distribution**

   * Use buffered `chan` of tasks for simple, lock‐free work sharing.
3. **Work Stealing (Optional)**

   * Implement a deque with `PushBottom`/`StealTop` to mitigate starvation and imbalance for large K.
4. **Cache‐Aligned Buffers**

   * Preallocate `AlignedBuffer` slices per worker to minimize GC and L1/L2 cache misses.
5. **Profiling Hooks**

   * Wrap each HE primitive to record durations/counts; disable detailed profiling for small workloads to avoid overhead.

### 5. Key Insights & Best Practices

* **Bottleneck Focus**: Target relinearization (key-switch) for algorithmic or assembly (SIMD/AVX) optimizations.
* **Memory Bandwidth**: Be aware of DRAM saturation; pin goroutines/data to NUMA nodes when possible.
* **Optimal Thread Count**: 2–4 threads for most workloads; use dynamic adjustment based on batch size.
* **Batch Size Matters**: Small K (<128) suffers profiling overhead—fall back to sequential or lightweight profiling.
* **Pipeline Parallelism**: For extreme throughput, stage Mul→Relin→Rescale→Rotate in a 4-stage pipeline with dedicated workers.

---

*This summary can be used by AI agents as a self‑contained reference for implementing, profiling, and optimizing parallel CKKS inner products in Lattigo.*
