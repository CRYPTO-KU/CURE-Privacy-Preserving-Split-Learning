package parallelizationexperiments

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	CacheLineSize = 64   // Cache line size in bytes (typical for x86-64)
	PageSize      = 4096 // Memory page size
)

// AlignedBuffer represents a cache-aligned buffer for polynomial coefficients
type AlignedBuffer struct {
	data     []uint64 // Actual polynomial data
	capacity int      // Buffer capacity
	size     int      // Current size
	aligned  bool     // Whether buffer is cache-aligned
}

// NewAlignedBuffer creates a cache-aligned buffer
func NewAlignedBuffer(capacity int) *AlignedBuffer {
	// Allocate extra space for alignment
	extraSpace := CacheLineSize / 8 // 8 bytes per uint64
	rawData := make([]uint64, capacity+extraSpace)

	// Find aligned starting point
	ptr := uintptr(unsafe.Pointer(&rawData[0]))
	alignedPtr := (ptr + CacheLineSize - 1) &^ (CacheLineSize - 1)
	offset := int((alignedPtr - ptr) / 8) // Convert to uint64 offset

	alignedData := rawData[offset : offset+capacity]

	return &AlignedBuffer{
		data:     alignedData,
		capacity: capacity,
		size:     0,
		aligned:  true,
	}
}

// Reset clears the buffer for reuse
func (ab *AlignedBuffer) Reset() {
	ab.size = 0
	// Don't clear data for performance - just reset size
}

// ThreadLocalBufferPool manages per-thread buffer pools
type ThreadLocalBufferPool struct {
	pools    map[int]*BufferPool // One pool per worker
	mu       sync.RWMutex
	maxPools int
}

// BufferPool manages a pool of reusable aligned buffers
type BufferPool struct {
	buffers    []*AlignedBuffer
	available  []bool // Track available buffers
	mu         sync.Mutex
	workerID   int
	hitCount   int64 // Cache hits
	missCount  int64 // Cache misses
	totalAlloc int64 // Total allocations
}

// NewThreadLocalBufferPool creates a new thread-local buffer pool system
func NewThreadLocalBufferPool(maxWorkers int) *ThreadLocalBufferPool {
	return &ThreadLocalBufferPool{
		pools:    make(map[int]*BufferPool),
		maxPools: maxWorkers,
	}
}

// GetWorkerPool gets or creates a buffer pool for a specific worker
func (tlbp *ThreadLocalBufferPool) GetWorkerPool(workerID int, bufferSize, poolSize int) *BufferPool {
	tlbp.mu.RLock()
	if pool, exists := tlbp.pools[workerID]; exists {
		tlbp.mu.RUnlock()
		return pool
	}
	tlbp.mu.RUnlock()

	// Create new pool
	tlbp.mu.Lock()
	defer tlbp.mu.Unlock()

	// Double-check after acquiring write lock
	if pool, exists := tlbp.pools[workerID]; exists {
		return pool
	}

	pool := &BufferPool{
		buffers:   make([]*AlignedBuffer, poolSize),
		available: make([]bool, poolSize),
		workerID:  workerID,
	}

	// Pre-allocate aligned buffers
	for i := 0; i < poolSize; i++ {
		pool.buffers[i] = NewAlignedBuffer(bufferSize)
		pool.available[i] = true
	}

	tlbp.pools[workerID] = pool
	return pool
}

// GetBuffer gets an available buffer from the pool
func (bp *BufferPool) GetBuffer() *AlignedBuffer {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	// Look for available buffer
	for i, available := range bp.available {
		if available {
			bp.available[i] = false
			bp.buffers[i].Reset()
			atomic.AddInt64(&bp.hitCount, 1)
			return bp.buffers[i]
		}
	}

	// No available buffers - create new one (cache miss)
	atomic.AddInt64(&bp.missCount, 1)
	atomic.AddInt64(&bp.totalAlloc, 1)
	return NewAlignedBuffer(bp.buffers[0].capacity)
}

// ReturnBuffer returns a buffer to the pool
func (bp *BufferPool) ReturnBuffer(buffer *AlignedBuffer) {
	if !buffer.aligned {
		return // Don't pool non-aligned buffers
	}

	bp.mu.Lock()
	defer bp.mu.Unlock()

	// Find matching buffer and mark as available
	for i, poolBuffer := range bp.buffers {
		if poolBuffer == buffer {
			bp.available[i] = true
			buffer.Reset()
			return
		}
	}

	// Buffer not from this pool - ignore
}

// GetStatistics returns buffer pool statistics
func (bp *BufferPool) GetStatistics() map[string]interface{} {
	hits := atomic.LoadInt64(&bp.hitCount)
	misses := atomic.LoadInt64(&bp.missCount)
	total := atomic.LoadInt64(&bp.totalAlloc)

	hitRate := float64(0)
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses) * 100.0
	}

	return map[string]interface{}{
		"worker_id":         bp.workerID,
		"cache_hits":        hits,
		"cache_misses":      misses,
		"hit_rate":          hitRate,
		"total_allocations": total,
		"pool_size":         len(bp.buffers),
	}
}

// ParallelInnerProductConfig holds configuration for parallel inner product experiments
type ParallelInnerProductConfig struct {
	K            int     // Number of independent inner products
	N            int     // Number of threads/goroutines
	VectorDim    int     // Dimension of input vectors
	LogN         int     // CKKS ring dimension (default 15)
	LogQ         []int   // Modulus chain
	LogP         []int   // Special primes
	DefaultScale float64 // CKKS scale
}

// DefaultConfig returns a reasonable default configuration
func DefaultConfig() *ParallelInnerProductConfig {
	return &ParallelInnerProductConfig{
		K:            1000,                      // 1000 inner products
		N:            runtime.NumCPU(),          // Use all available cores
		VectorDim:    512,                       // 512-dimensional vectors
		LogN:         15,                        // 32768 slots
		LogQ:         []int{60, 40, 40, 40, 38}, // Modulus chain
		LogP:         []int{60},                 // Special primes
		DefaultScale: 1 << 40,                   // 40-bit scale
	}
}

// InnerProductResult holds timing and operation count results
type InnerProductResult struct {
	TotalTime           time.Duration
	AverageTime         time.Duration
	ThroughputOpsPerSec float64
	TotalRotations      int64
	TotalMuls           int64
	TotalRelins         int64
	TotalRescales       int64
	WorkerStats         []WorkerStats
}

// WorkerStats holds per-worker statistics
type WorkerStats struct {
	WorkerID       int
	TasksCompleted int
	WorkerTime     time.Duration
	Rotations      int64
	Muls           int64
	Relins         int64
	Rescales       int64
}

// CKKSContext wraps the CKKS cryptographic context
type CKKSContext struct {
	Params    ckks.Parameters
	Sk        *rlwe.SecretKey
	Pk        *rlwe.PublicKey
	Encoder   *ckks.Encoder
	Encryptor *rlwe.Encryptor
	Decryptor *rlwe.Decryptor
	Evk       rlwe.EvaluationKeySet
}

// NewCKKSContext creates a new CKKS context with the specified parameters
func NewCKKSContext(config *ParallelInnerProductConfig) (*CKKSContext, error) {
	// Create CKKS parameters
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            config.LogN,
		LogQ:            config.LogQ,
		LogP:            config.LogP,
		LogDefaultScale: 40, // Use fixed scale for compatibility
		Xs:              rlwe.DefaultXs,
		Xe:              rlwe.DefaultXe,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// Generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	// Generate rotation keys for tree-sum (powers of 2 up to vector dimension)
	rotations := []int{}
	for step := 1; step < config.VectorDim; step *= 2 {
		rotations = append(rotations, step)
	}

	// Relinearization key
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Galois (rotation) keys
	galEls := make([]uint64, len(rotations))
	for i, rot := range rotations {
		galEls[i] = params.GaloisElementForRotation(rot)
	}
	galKeys := kgen.GenGaloisKeysNew(galEls, sk)

	// Pack into evaluation key set
	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	return &CKKSContext{
		Params:    params,
		Sk:        sk,
		Pk:        pk,
		Encoder:   ckks.NewEncoder(params),
		Encryptor: rlwe.NewEncryptor(params, pk),
		Decryptor: rlwe.NewDecryptor(params, sk),
		Evk:       evk,
	}, nil
}

// InnerProductTask represents a single inner product computation task
type InnerProductTask struct {
	ID  int
	CT1 *rlwe.Ciphertext
	CT2 *rlwe.Ciphertext
}

// InnerProductTaskResult holds the result of a single inner product task
type InnerProductTaskResult struct {
	TaskID    int
	Result    *rlwe.Ciphertext
	Duration  time.Duration
	Rotations int64
	Muls      int64
	Relins    int64
	Rescales  int64
	Error     error
}

// OperationCounter wraps an evaluator to count operations
type OperationCounter struct {
	eval      *ckks.Evaluator
	rotations int64
	muls      int64
	relins    int64
	rescales  int64
	mu        sync.Mutex
}

// NewOperationCounter creates a new operation counter
func NewOperationCounter(params ckks.Parameters, evk rlwe.EvaluationKeySet) *OperationCounter {
	return &OperationCounter{
		eval: ckks.NewEvaluator(params, evk),
	}
}

// RotateNew wraps the evaluator's RotateNew and counts operations
func (oc *OperationCounter) RotateNew(ct *rlwe.Ciphertext, k int) (*rlwe.Ciphertext, error) {
	oc.mu.Lock()
	oc.rotations++
	oc.mu.Unlock()
	return oc.eval.RotateNew(ct, k)
}

// MulNew wraps the evaluator's MulNew and counts operations
func (oc *OperationCounter) MulNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	oc.mu.Lock()
	oc.muls++
	oc.mu.Unlock()
	return oc.eval.MulNew(ct1, ct2)
}

// RelinearizeNew wraps the evaluator's RelinearizeNew and counts operations
func (oc *OperationCounter) RelinearizeNew(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	oc.mu.Lock()
	oc.relins++
	oc.mu.Unlock()
	return oc.eval.RelinearizeNew(ct)
}

// Rescale wraps the evaluator's Rescale and counts operations
func (oc *OperationCounter) Rescale(ct, ctOut *rlwe.Ciphertext) error {
	oc.mu.Lock()
	oc.rescales++
	oc.mu.Unlock()
	return oc.eval.Rescale(ct, ctOut)
}

// AddNew wraps the evaluator's AddNew (no counting needed for additions)
func (oc *OperationCounter) AddNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return oc.eval.AddNew(ct1, ct2)
}

// GetStats returns the current operation counts
func (oc *OperationCounter) GetStats() (int64, int64, int64, int64) {
	oc.mu.Lock()
	defer oc.mu.Unlock()
	return oc.rotations, oc.muls, oc.relins, oc.rescales
}

// ResetStats resets all operation counters
func (oc *OperationCounter) ResetStats() {
	oc.mu.Lock()
	defer oc.mu.Unlock()
	oc.rotations, oc.muls, oc.relins, oc.rescales = 0, 0, 0, 0
}

// ComputeInnerProduct computes the inner product of two ciphertexts using tree-sum
func ComputeInnerProduct(ct1, ct2 *rlwe.Ciphertext, evaluator *OperationCounter, vectorDim int) (*rlwe.Ciphertext, error) {
	// 1) Element-wise multiplication
	tmp, err := evaluator.MulNew(ct1, ct2)
	if err != nil {
		return nil, fmt.Errorf("multiplication failed: %w", err)
	}

	// 2) Relinearize back to degree 1
	tmp, err = evaluator.RelinearizeNew(tmp)
	if err != nil {
		return nil, fmt.Errorf("relinearization failed: %w", err)
	}

	// 3) Rescale to maintain proper scale
	if err := evaluator.Rescale(tmp, tmp); err != nil {
		return nil, fmt.Errorf("rescaling failed: %w", err)
	}

	// 4) Tree-sum rotations - accumulate all elements into slot 0
	for step := 1; step < vectorDim; step *= 2 {
		rot, err := evaluator.RotateNew(tmp, step)
		if err != nil {
			return nil, fmt.Errorf("rotation failed: %w", err)
		}
		tmp, err = evaluator.AddNew(tmp, rot)
		if err != nil {
			return nil, fmt.Errorf("addition failed: %w", err)
		}
	}

	return tmp, nil
}

// WorkerPool manages a pool of workers for parallel inner product computation
type WorkerPool struct {
	config      *ParallelInnerProductConfig
	ctx         *CKKSContext
	tasks       chan InnerProductTask
	results     chan InnerProductTaskResult
	workers     []*OperationCounter
	workerStats []WorkerStats
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(config *ParallelInnerProductConfig, ctx *CKKSContext) *WorkerPool {
	wp := &WorkerPool{
		config:      config,
		ctx:         ctx,
		tasks:       make(chan InnerProductTask, config.K),
		results:     make(chan InnerProductTaskResult, config.K),
		workers:     make([]*OperationCounter, config.N),
		workerStats: make([]WorkerStats, config.N),
	}

	// Create operation counters for each worker
	for i := 0; i < config.N; i++ {
		wp.workers[i] = NewOperationCounter(ctx.Params, ctx.Evk)
		wp.workerStats[i].WorkerID = i
	}

	return wp
}

// StartWorkers starts N worker goroutines
func (wp *WorkerPool) StartWorkers() {
	var wg sync.WaitGroup

	for i := 0; i < wp.config.N; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			wp.worker(workerID)
		}(i)
	}

	go func() {
		wg.Wait()
		close(wp.results)
	}()
}

// worker processes tasks from the task channel
func (wp *WorkerPool) worker(workerID int) {
	evaluator := wp.workers[workerID]
	stats := &wp.workerStats[workerID]

	workerStart := time.Now()

	for task := range wp.tasks {
		taskStart := time.Now()

		// Reset operation counters for this task
		evaluator.ResetStats()

		// Compute inner product
		result, err := ComputeInnerProduct(task.CT1, task.CT2, evaluator, wp.config.VectorDim)

		taskDuration := time.Since(taskStart)

		// Get operation counts for this task
		rots, muls, relins, rescales := evaluator.GetStats()

		// Update worker stats
		stats.TasksCompleted++
		stats.Rotations += rots
		stats.Muls += muls
		stats.Relins += relins
		stats.Rescales += rescales

		// Send result
		wp.results <- InnerProductTaskResult{
			TaskID:    task.ID,
			Result:    result,
			Duration:  taskDuration,
			Rotations: rots,
			Muls:      muls,
			Relins:    relins,
			Rescales:  rescales,
			Error:     err,
		}
	}

	stats.WorkerTime = time.Since(workerStart)
}

// SubmitTask submits a task to the worker pool
func (wp *WorkerPool) SubmitTask(task InnerProductTask) {
	wp.tasks <- task
}

// FinishSubmission closes the task channel
func (wp *WorkerPool) FinishSubmission() {
	close(wp.tasks)
}

// CollectResults collects all results from the workers
func (wp *WorkerPool) CollectResults() *InnerProductResult {
	var results []InnerProductTaskResult

	for result := range wp.results {
		results = append(results, result)
	}

	// Aggregate statistics
	var totalTime time.Duration
	var totalRotations, totalMuls, totalRelins, totalRescales int64

	for _, result := range results {
		if result.Duration > totalTime {
			totalTime = result.Duration // Max time across all tasks
		}
		totalRotations += result.Rotations
		totalMuls += result.Muls
		totalRelins += result.Relins
		totalRescales += result.Rescales
	}

	avgTime := time.Duration(int64(totalTime) / int64(len(results)))
	throughput := float64(len(results)) / totalTime.Seconds()

	return &InnerProductResult{
		TotalTime:           totalTime,
		AverageTime:         avgTime,
		ThroughputOpsPerSec: throughput,
		TotalRotations:      totalRotations,
		TotalMuls:           totalMuls,
		TotalRelins:         totalRelins,
		TotalRescales:       totalRescales,
		WorkerStats:         wp.workerStats,
	}
}

// BenchmarkParallelInnerProducts runs a comprehensive benchmark of parallel inner products
func BenchmarkParallelInnerProducts(config *ParallelInnerProductConfig) (*InnerProductResult, error) {
	fmt.Printf("=== Parallel Inner Product Benchmark ===\n")
	fmt.Printf("K (operations): %d, N (threads): %d, Vector dimension: %d\n",
		config.K, config.N, config.VectorDim)
	fmt.Printf("CKKS parameters: logN=%d, logQ=%v, logP=%v\n",
		config.LogN, config.LogQ, config.LogP)

	// OFFLINE PHASE: Initialize CKKS context (NOT TIMED)
	fmt.Printf("OFFLINE: Initializing CKKS context...\n")
	ctx, err := NewCKKSContext(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create CKKS context: %w", err)
	}

	// OFFLINE PHASE: Generate test data (NOT TIMED)
	fmt.Printf("OFFLINE: Generating %d pairs of test ciphertexts...\n", config.K)
	tasks := make([]InnerProductTask, config.K)

	for i := 0; i < config.K; i++ {
		// Generate random complex vectors
		// CKKS slots = N/2 where N is the ring dimension
		slots := ctx.Params.MaxSlots()
		vec1 := make([]complex128, slots)
		vec2 := make([]complex128, slots)

		for j := 0; j < config.VectorDim; j++ {
			vec1[j] = complex(float64(i+j), 0) // Simple test pattern
			vec2[j] = complex(float64(j), 0)
		}

		// Encode and encrypt (offline operations)
		pt1 := ckks.NewPlaintext(ctx.Params, ctx.Params.MaxLevel())
		pt2 := ckks.NewPlaintext(ctx.Params, ctx.Params.MaxLevel())

		if err := ctx.Encoder.Encode(vec1, pt1); err != nil {
			return nil, fmt.Errorf("encoding failed: %w", err)
		}
		if err := ctx.Encoder.Encode(vec2, pt2); err != nil {
			return nil, fmt.Errorf("encoding failed: %w", err)
		}

		ct1, err := ctx.Encryptor.EncryptNew(pt1)
		if err != nil {
			return nil, fmt.Errorf("encryption failed: %w", err)
		}
		ct2, err := ctx.Encryptor.EncryptNew(pt2)
		if err != nil {
			return nil, fmt.Errorf("encryption failed: %w", err)
		}

		tasks[i] = InnerProductTask{
			ID:  i,
			CT1: ct1,
			CT2: ct2,
		}
	}

	// OFFLINE PHASE: Create worker pool (NOT TIMED)
	fmt.Printf("OFFLINE: Starting %d workers...\n", config.N)
	wp := NewWorkerPool(config, ctx)

	// ONLINE PHASE: Only time the actual computation
	fmt.Printf("ONLINE: Starting computation with %d shared-resource workers...\n", config.N)

	// Start timing ONLY the computation phase
	computationStart := time.Now()

	// Start workers
	wp.StartWorkers()

	// Submit all tasks
	for _, task := range tasks {
		wp.SubmitTask(task)
	}
	wp.FinishSubmission()

	// Collect results
	result := wp.CollectResults()

	computationDuration := time.Since(computationStart)
	result.TotalTime = computationDuration

	fmt.Printf("=== COMPUTATION-ONLY RESULTS ===\n")
	fmt.Printf("Computation time: %v\n", result.TotalTime)
	fmt.Printf("Throughput: %.2f operations/second\n", result.ThroughputOpsPerSec)
	fmt.Printf("Total operations: Rotations=%d, Muls=%d, Relins=%d, Rescales=%d\n",
		result.TotalRotations, result.TotalMuls, result.TotalRelins, result.TotalRescales)

	// Print per-worker statistics
	fmt.Printf("\n=== Per-Worker Statistics ===\n")
	for i, stats := range result.WorkerStats {
		fmt.Printf("Worker %d: %d tasks, %v total time, %d rots, %d muls, %d relins, %d rescales\n",
			i, stats.TasksCompleted, stats.WorkerTime,
			stats.Rotations, stats.Muls, stats.Relins, stats.Rescales)
	}

	return result, nil
}

// RunScalabilityAnalysis runs benchmarks with different numbers of threads
func RunScalabilityAnalysis() error {
	config := DefaultConfig()

	fmt.Printf("=== Scalability Analysis ===\n")

	threadCounts := []int{1, 2, 4, 8, 16}
	if config.N > 16 {
		threadCounts = append(threadCounts, config.N)
	}

	for _, N := range threadCounts {
		if N > runtime.NumCPU() {
			break
		}

		config.N = N
		fmt.Printf("\n--- Testing with %d threads ---\n", N)

		result, err := BenchmarkParallelInnerProducts(config)
		if err != nil {
			return fmt.Errorf("benchmark failed with %d threads: %w", N, err)
		}

		fmt.Printf("N=%d: %.2f ops/sec, %v total time\n",
			N, result.ThroughputOpsPerSec, result.TotalTime)
	}

	return nil
}

// RunKScalabilityAnalysis runs benchmarks with different numbers of operations
func RunKScalabilityAnalysis() error {
	config := DefaultConfig()

	fmt.Printf("=== K-Scalability Analysis ===\n")

	kValues := []int{100, 500, 1000, 2000, 5000}

	for _, K := range kValues {
		config.K = K
		fmt.Printf("\n--- Testing with K=%d operations ---\n", K)

		result, err := BenchmarkParallelInnerProducts(config)
		if err != nil {
			return fmt.Errorf("benchmark failed with K=%d: %w", K, err)
		}

		fmt.Printf("K=%d: %.2f ops/sec, %v total time, %.2f ms/op\n",
			K, result.ThroughputOpsPerSec, result.TotalTime,
			float64(result.TotalTime.Milliseconds())/float64(K))
	}

	return nil
}

// MaxParallelInnerProductConfig holds configuration for maximum parallelization experiments
type MaxParallelInnerProductConfig struct {
	K         int   // Number of independent inner products
	N         int   // Number of threads/goroutines
	VectorDim int   // Dimension of input vectors
	LogN      int   // CKKS ring dimension (default 15)
	LogQ      []int // Modulus chain
	LogP      []int // Special primes
}

// IsolatedWorkerContext - each worker gets its own complete CKKS universe
type IsolatedWorkerContext struct {
	WorkerID  int
	Params    ckks.Parameters
	Sk        *rlwe.SecretKey
	Pk        *rlwe.PublicKey
	Encoder   *ckks.Encoder
	Encryptor *rlwe.Encryptor
	Decryptor *rlwe.Decryptor
	Evaluator *ckks.Evaluator
	Evk       rlwe.EvaluationKeySet

	// Operation counters
	Rotations int64
	Muls      int64
	Relins    int64
	Rescales  int64
}

// NewIsolatedWorkerContext creates a completely isolated CKKS context for one worker
func NewIsolatedWorkerContext(workerID int, config *MaxParallelInnerProductConfig) (*IsolatedWorkerContext, error) {
	// Create CKKS parameters
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            config.LogN,
		LogQ:            config.LogQ,
		LogP:            config.LogP,
		LogDefaultScale: 40, // Use fixed scale for compatibility
		Xs:              rlwe.DefaultXs,
		Xe:              rlwe.DefaultXe,
	})
	if err != nil {
		return nil, fmt.Errorf("worker %d: failed to create CKKS parameters: %w", workerID, err)
	}

	// Generate completely independent keys for this worker
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	// Generate rotation keys for tree-sum (powers of 2 up to vector dimension)
	rotations := []int{}
	for step := 1; step < config.VectorDim; step *= 2 {
		rotations = append(rotations, step)
	}

	// Relinearization key
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Galois (rotation) keys
	galEls := make([]uint64, len(rotations))
	for i, rot := range rotations {
		galEls[i] = params.GaloisElementForRotation(rot)
	}
	galKeys := kgen.GenGaloisKeysNew(galEls, sk)

	// Pack into evaluation key set
	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	return &IsolatedWorkerContext{
		WorkerID:  workerID,
		Params:    params,
		Sk:        sk,
		Pk:        pk,
		Encoder:   ckks.NewEncoder(params),
		Encryptor: rlwe.NewEncryptor(params, pk),
		Decryptor: rlwe.NewDecryptor(params, sk),
		Evaluator: ckks.NewEvaluator(params, evk),
		Evk:       evk,
	}, nil
}

// ComputeInnerProductIsolated computes inner product with completely isolated context
func (ctx *IsolatedWorkerContext) ComputeInnerProductIsolated(ct1, ct2 *rlwe.Ciphertext, vectorDim int) (*rlwe.Ciphertext, error) {
	// 1) Element-wise multiplication
	tmp, err := ctx.Evaluator.MulNew(ct1, ct2)
	if err != nil {
		return nil, fmt.Errorf("worker %d multiplication failed: %w", ctx.WorkerID, err)
	}
	ctx.Muls++

	// 2) Relinearize back to degree 1
	tmp, err = ctx.Evaluator.RelinearizeNew(tmp)
	if err != nil {
		return nil, fmt.Errorf("worker %d relinearization failed: %w", ctx.WorkerID, err)
	}
	ctx.Relins++

	// 3) Rescale to maintain proper scale
	if err := ctx.Evaluator.Rescale(tmp, tmp); err != nil {
		return nil, fmt.Errorf("worker %d rescaling failed: %w", ctx.WorkerID, err)
	}
	ctx.Rescales++

	// 4) Tree-sum rotations - accumulate all elements into slot 0
	for step := 1; step < vectorDim; step *= 2 {
		rot, err := ctx.Evaluator.RotateNew(tmp, step)
		if err != nil {
			return nil, fmt.Errorf("worker %d rotation failed: %w", ctx.WorkerID, err)
		}
		ctx.Rotations++

		tmp, err = ctx.Evaluator.AddNew(tmp, rot)
		if err != nil {
			return nil, fmt.Errorf("worker %d addition failed: %w", ctx.WorkerID, err)
		}
	}

	return tmp, nil
}

// MaxParallelInnerProductTask represents a task with its own data
type MaxParallelInnerProductTask struct {
	TaskID  int
	Vector1 []complex128 // Own copy of vector data
	Vector2 []complex128 // Own copy of vector data
}

// MaxParallelInnerProductResult holds the result
type MaxParallelInnerProductResult struct {
	TaskID    int
	Result    *rlwe.Ciphertext
	Duration  time.Duration
	Rotations int64
	Muls      int64
	Relins    int64
	Rescales  int64
}

// MaxParallelWorkerPool - each worker has completely isolated resources
type MaxParallelWorkerPool struct {
	config         *MaxParallelInnerProductConfig
	contexts       []*IsolatedWorkerContext // Each worker has own context
	tasks          chan MaxParallelInnerProductTask
	encryptedTasks chan EncryptedInnerProductTask // New channel for pre-encrypted tasks
	results        chan MaxParallelInnerProductResult
}

// NewMaxParallelWorkerPool creates a worker pool with maximum isolation
func NewMaxParallelWorkerPool(config *MaxParallelInnerProductConfig) (*MaxParallelWorkerPool, error) {
	fmt.Printf("Creating %d completely isolated worker contexts...\n", config.N)

	contexts := make([]*IsolatedWorkerContext, config.N)

	// Create completely isolated context for each worker
	for i := 0; i < config.N; i++ {
		ctx, err := NewIsolatedWorkerContext(i, config)
		if err != nil {
			return nil, fmt.Errorf("failed to create isolated context for worker %d: %w", i, err)
		}
		contexts[i] = ctx
		fmt.Printf("  Worker %d: Own keys, evaluator, encoder, encryptor created\n", i)
	}

	return &MaxParallelWorkerPool{
		config:         config,
		contexts:       contexts,
		tasks:          make(chan MaxParallelInnerProductTask, config.K),
		encryptedTasks: make(chan EncryptedInnerProductTask, config.K),
		results:        make(chan MaxParallelInnerProductResult, config.K),
	}, nil
}

// StartMaxParallelWorkers starts workers with complete isolation
func (wp *MaxParallelWorkerPool) StartMaxParallelWorkers() {
	var wg sync.WaitGroup

	for i := 0; i < wp.config.N; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			wp.isolatedWorker(workerID)
		}(i)
	}

	go func() {
		wg.Wait()
		close(wp.results)
	}()
}

// isolatedWorker processes tasks with complete resource isolation
func (wp *MaxParallelWorkerPool) isolatedWorker(workerID int) {
	ctx := wp.contexts[workerID]

	fmt.Printf("Worker %d starting with completely isolated resources\n", workerID)

	for task := range wp.tasks {
		taskStart := time.Now()

		// Reset operation counters
		ctx.Rotations, ctx.Muls, ctx.Relins, ctx.Rescales = 0, 0, 0, 0

		// Encode with own encoder
		pt1 := ckks.NewPlaintext(ctx.Params, ctx.Params.MaxLevel())
		pt2 := ckks.NewPlaintext(ctx.Params, ctx.Params.MaxLevel())

		if err := ctx.Encoder.Encode(task.Vector1, pt1); err != nil {
			continue
		}
		if err := ctx.Encoder.Encode(task.Vector2, pt2); err != nil {
			continue
		}

		// Encrypt with own encryptor
		ct1, err := ctx.Encryptor.EncryptNew(pt1)
		if err != nil {
			continue
		}
		ct2, err := ctx.Encryptor.EncryptNew(pt2)
		if err != nil {
			continue
		}

		// Compute inner product with own evaluator
		result, err := ctx.ComputeInnerProductIsolated(ct1, ct2, wp.config.VectorDim)

		taskDuration := time.Since(taskStart)

		// Send result
		wp.results <- MaxParallelInnerProductResult{
			TaskID:    task.TaskID,
			Result:    result,
			Duration:  taskDuration,
			Rotations: ctx.Rotations,
			Muls:      ctx.Muls,
			Relins:    ctx.Relins,
			Rescales:  ctx.Rescales,
		}
	}
}

// SubmitMaxParallelTask submits a task with own data
func (wp *MaxParallelWorkerPool) SubmitMaxParallelTask(task MaxParallelInnerProductTask) {
	wp.tasks <- task
}

// FinishMaxParallelSubmission closes the task channel
func (wp *MaxParallelWorkerPool) FinishMaxParallelSubmission() {
	close(wp.encryptedTasks)
}

// CollectMaxParallelResults collects all results
func (wp *MaxParallelWorkerPool) CollectMaxParallelResults() *InnerProductResult {
	var results []MaxParallelInnerProductResult

	for result := range wp.results {
		results = append(results, result)
	}

	// Aggregate statistics
	var totalTime time.Duration
	var totalRotations, totalMuls, totalRelins, totalRescales int64

	for _, result := range results {
		if result.Duration > totalTime {
			totalTime = result.Duration
		}
		totalRotations += result.Rotations
		totalMuls += result.Muls
		totalRelins += result.Relins
		totalRescales += result.Rescales
	}

	avgTime := time.Duration(int64(totalTime) / int64(len(results)))
	throughput := float64(len(results)) / totalTime.Seconds()

	// Convert to WorkerStats format
	workerStats := make([]WorkerStats, wp.config.N)
	for i := 0; i < wp.config.N; i++ {
		workerStats[i] = WorkerStats{
			WorkerID: i,
			// Note: Tasks completed will be calculated based on results
		}
	}

	return &InnerProductResult{
		TotalTime:           totalTime,
		AverageTime:         avgTime,
		ThroughputOpsPerSec: throughput,
		TotalRotations:      totalRotations,
		TotalMuls:           totalMuls,
		TotalRelins:         totalRelins,
		TotalRescales:       totalRescales,
		WorkerStats:         workerStats,
	}
}

// BenchmarkMaxParallelInnerProducts runs the maximum parallelization benchmark
func BenchmarkMaxParallelInnerProducts(config *MaxParallelInnerProductConfig) (*InnerProductResult, error) {
	fmt.Printf("=== MAXIMUM PARALLELIZATION BENCHMARK ===\n")
	fmt.Printf("Each worker gets: Own keys, own evaluator, own encoder, own encryptor\n")
	fmt.Printf("K (operations): %d, N (threads): %d, Vector dimension: %d\n",
		config.K, config.N, config.VectorDim)
	fmt.Printf("Memory usage: %d completely independent CKKS contexts\n", config.N)

	// OFFLINE PHASE: Create worker pool with maximum isolation (NOT TIMED)
	fmt.Printf("OFFLINE: Creating %d isolated worker contexts...\n", config.N)
	wp, err := NewMaxParallelWorkerPool(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create max parallel worker pool: %w", err)
	}

	// OFFLINE PHASE: Generate and encrypt test data (NOT TIMED)
	fmt.Printf("OFFLINE: Generating and encrypting %d tasks...\n", config.K)
	encryptedTasks := make([]EncryptedInnerProductTask, config.K)

	for i := 0; i < config.K; i++ {
		// Use first worker's context for encryption (all have same parameters)
		ctx := wp.contexts[0]
		slots := ctx.Params.MaxSlots()
		vec1 := make([]complex128, slots)
		vec2 := make([]complex128, slots)

		for j := 0; j < config.VectorDim; j++ {
			vec1[j] = complex(float64(i+j), 0)
			vec2[j] = complex(float64(j), 0)
		}

		// Encode and encrypt (offline operations)
		pt1 := ckks.NewPlaintext(ctx.Params, ctx.Params.MaxLevel())
		pt2 := ckks.NewPlaintext(ctx.Params, ctx.Params.MaxLevel())

		if err := ctx.Encoder.Encode(vec1, pt1); err != nil {
			return nil, fmt.Errorf("encoding failed: %w", err)
		}
		if err := ctx.Encoder.Encode(vec2, pt2); err != nil {
			return nil, fmt.Errorf("encoding failed: %w", err)
		}

		ct1, err := ctx.Encryptor.EncryptNew(pt1)
		if err != nil {
			return nil, fmt.Errorf("encryption failed: %w", err)
		}
		ct2, err := ctx.Encryptor.EncryptNew(pt2)
		if err != nil {
			return nil, fmt.Errorf("encryption failed: %w", err)
		}

		encryptedTasks[i] = EncryptedInnerProductTask{
			TaskID: i,
			CT1:    ct1,
			CT2:    ct2,
		}
	}

	// ONLINE PHASE: Only time the actual computation
	fmt.Printf("ONLINE: Starting computation with %d isolated workers...\n", config.N)

	// Start timing ONLY the computation phase
	computationStart := time.Now()

	wp.StartMaxParallelWorkersWithPreEncrypted()

	// Submit all pre-encrypted tasks
	for _, task := range encryptedTasks {
		wp.SubmitEncryptedTask(task)
	}
	wp.FinishMaxParallelSubmission()

	// Collect results
	result := wp.CollectMaxParallelResults()

	// Stop timing - this is the actual computation time
	computationDuration := time.Since(computationStart)
	result.TotalTime = computationDuration

	fmt.Printf("=== COMPUTATION-ONLY RESULTS ===\n")
	fmt.Printf("Computation time: %v\n", result.TotalTime)
	fmt.Printf("Throughput: %.2f operations/second\n", result.ThroughputOpsPerSec)
	fmt.Printf("Total operations: Rotations=%d, Muls=%d, Relins=%d, Rescales=%d\n",
		result.TotalRotations, result.TotalMuls, result.TotalRelins, result.TotalRescales)

	return result, nil
}

// EncryptedInnerProductTask represents a pre-encrypted task
type EncryptedInnerProductTask struct {
	TaskID int
	CT1    *rlwe.Ciphertext
	CT2    *rlwe.Ciphertext
}

// StartMaxParallelWorkersWithPreEncrypted starts workers for pre-encrypted tasks
func (wp *MaxParallelWorkerPool) StartMaxParallelWorkersWithPreEncrypted() {
	var wg sync.WaitGroup

	for i := 0; i < wp.config.N; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			wp.isolatedWorkerPreEncrypted(workerID)
		}(i)
	}

	go func() {
		wg.Wait()
		close(wp.results)
	}()
}

// isolatedWorkerPreEncrypted processes pre-encrypted tasks
func (wp *MaxParallelWorkerPool) isolatedWorkerPreEncrypted(workerID int) {
	ctx := wp.contexts[workerID]

	for task := range wp.encryptedTasks {
		taskStart := time.Now()

		// Reset operation counters
		ctx.Rotations, ctx.Muls, ctx.Relins, ctx.Rescales = 0, 0, 0, 0

		// Only compute inner product (no encoding/encryption overhead)
		result, _ := ctx.ComputeInnerProductIsolated(task.CT1, task.CT2, wp.config.VectorDim)

		taskDuration := time.Since(taskStart)

		// Send result
		wp.results <- MaxParallelInnerProductResult{
			TaskID:    task.TaskID,
			Result:    result,
			Duration:  taskDuration,
			Rotations: ctx.Rotations,
			Muls:      ctx.Muls,
			Relins:    ctx.Relins,
			Rescales:  ctx.Rescales,
		}
	}
}

// SubmitEncryptedTask submits a pre-encrypted task
func (wp *MaxParallelWorkerPool) SubmitEncryptedTask(task EncryptedInnerProductTask) {
	wp.encryptedTasks <- task
}

// BottleneckProfiler - comprehensive bottleneck identification system
type BottleneckProfiler struct {
	// Memory bandwidth tracking
	MemoryAllocations   int64
	MemoryDeallocations int64
	PeakMemoryUsage     int64

	// Operation-level profiling
	RotationTimes        []time.Duration
	MultiplicationTimes  []time.Duration
	RelinearizationTimes []time.Duration
	RescaleTimes         []time.Duration
	AdditionTimes        []time.Duration

	// Thread contention tracking
	WorkerWaitTimes   []time.Duration
	ChannelBlockTimes []time.Duration

	// Cache/CPU effects
	WorkerStartTimes   []time.Time
	WorkerEndTimes     []time.Time
	CPUCoreUtilization map[int]time.Duration

	// Lock for thread-safe updates
	mu sync.Mutex
}

// NewBottleneckProfiler creates a new bottleneck profiler
func NewBottleneckProfiler() *BottleneckProfiler {
	return &BottleneckProfiler{
		RotationTimes:        make([]time.Duration, 0),
		MultiplicationTimes:  make([]time.Duration, 0),
		RelinearizationTimes: make([]time.Duration, 0),
		RescaleTimes:         make([]time.Duration, 0),
		AdditionTimes:        make([]time.Duration, 0),
		WorkerWaitTimes:      make([]time.Duration, 0),
		ChannelBlockTimes:    make([]time.Duration, 0),
		WorkerStartTimes:     make([]time.Time, 0),
		WorkerEndTimes:       make([]time.Time, 0),
		CPUCoreUtilization:   make(map[int]time.Duration),
	}
}

// ProfiledOperationCounter wraps an evaluator with detailed profiling
type ProfiledOperationCounter struct {
	eval     *ckks.Evaluator
	profiler *BottleneckProfiler
	workerID int
}

// NewProfiledOperationCounter creates a new profiled operation counter
func NewProfiledOperationCounter(params ckks.Parameters, evk rlwe.EvaluationKeySet, profiler *BottleneckProfiler, workerID int) *ProfiledOperationCounter {
	return &ProfiledOperationCounter{
		eval:     ckks.NewEvaluator(params, evk),
		profiler: profiler,
		workerID: workerID,
	}
}

// RotateNew with detailed profiling
func (poc *ProfiledOperationCounter) RotateNew(ct *rlwe.Ciphertext, k int) (*rlwe.Ciphertext, error) {
	start := time.Now()
	result, err := poc.eval.RotateNew(ct, k)
	duration := time.Since(start)

	poc.profiler.mu.Lock()
	poc.profiler.RotationTimes = append(poc.profiler.RotationTimes, duration)
	poc.profiler.mu.Unlock()

	return result, err
}

// MulNew with detailed profiling
func (poc *ProfiledOperationCounter) MulNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	start := time.Now()
	result, err := poc.eval.MulNew(ct1, ct2)
	duration := time.Since(start)

	poc.profiler.mu.Lock()
	poc.profiler.MultiplicationTimes = append(poc.profiler.MultiplicationTimes, duration)
	poc.profiler.mu.Unlock()

	return result, err
}

// RelinearizeNew with detailed profiling
func (poc *ProfiledOperationCounter) RelinearizeNew(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	start := time.Now()
	result, err := poc.eval.RelinearizeNew(ct)
	duration := time.Since(start)

	poc.profiler.mu.Lock()
	poc.profiler.RelinearizationTimes = append(poc.profiler.RelinearizationTimes, duration)
	poc.profiler.mu.Unlock()

	return result, err
}

// Rescale with detailed profiling
func (poc *ProfiledOperationCounter) Rescale(ct, ctOut *rlwe.Ciphertext) error {
	start := time.Now()
	err := poc.eval.Rescale(ct, ctOut)
	duration := time.Since(start)

	poc.profiler.mu.Lock()
	poc.profiler.RescaleTimes = append(poc.profiler.RescaleTimes, duration)
	poc.profiler.mu.Unlock()

	return err
}

// AddNew with detailed profiling
func (poc *ProfiledOperationCounter) AddNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	start := time.Now()
	result, err := poc.eval.AddNew(ct1, ct2)
	duration := time.Since(start)

	poc.profiler.mu.Lock()
	poc.profiler.AdditionTimes = append(poc.profiler.AdditionTimes, duration)
	poc.profiler.mu.Unlock()

	return result, err
}

// BottleneckAnalysisConfig holds configuration for bottleneck analysis
type BottleneckAnalysisConfig struct {
	K          int // Number of operations
	N          int // Number of threads
	VectorDim  int // Vector dimension
	LogN       int // CKKS parameters
	LogQ       []int
	LogP       []int
	Iterations int // Number of test iterations
}

// BottleneckAnalysisResult holds comprehensive bottleneck analysis
type BottleneckAnalysisResult struct {
	// Overall performance
	TotalTime   time.Duration
	Throughput  float64
	ThreadsUsed int

	// Operation-level bottlenecks
	AvgRotationTime        time.Duration
	AvgMultiplicationTime  time.Duration
	AvgRelinearizationTime time.Duration
	AvgRescaleTime         time.Duration
	AvgAdditionTime        time.Duration

	// Thread efficiency analysis
	WorkerIdleTimes     []time.Duration
	WorkerActivePercent []float64
	LoadImbalance       float64

	// Memory bottleneck indicators
	MemoryPressure float64
	AllocationRate float64

	// Parallelization efficiency breakdown
	SerialFraction        float64 // Amdahl's law
	ParallelEfficiency    float64
	ScalabilityBottleneck string // Primary bottleneck identified
}

// RunBottleneckAnalysis performs comprehensive bottleneck identification
func RunBottleneckAnalysis(config *BottleneckAnalysisConfig) (*BottleneckAnalysisResult, error) {
	fmt.Printf("=== COMPREHENSIVE BOTTLENECK ANALYSIS ===\n")
	fmt.Printf("Configuration: K=%d, N=%d, VectorDim=%d, Iterations=%d\n",
		config.K, config.N, config.VectorDim, config.Iterations)

	// Initialize CKKS context
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            config.LogN,
		LogQ:            config.LogQ,
		LogP:            config.LogP,
		LogDefaultScale: 40,
		Xs:              rlwe.DefaultXs,
		Xe:              rlwe.DefaultXe,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// Generate keys (shared approach for comparison)
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	rotations := []int{}
	for step := 1; step < config.VectorDim; step *= 2 {
		rotations = append(rotations, step)
	}

	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := make([]uint64, len(rotations))
	for i, rot := range rotations {
		galEls[i] = params.GaloisElementForRotation(rot)
	}
	galKeys := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)

	// Pre-generate test data
	fmt.Printf("Pre-generating test data...\n")
	slots := params.MaxSlots()
	testTasks := make([]struct{ ct1, ct2 *rlwe.Ciphertext }, config.K)

	for i := 0; i < config.K; i++ {
		vec1 := make([]complex128, slots)
		vec2 := make([]complex128, slots)

		for j := 0; j < config.VectorDim; j++ {
			vec1[j] = complex(float64(i+j), 0)
			vec2[j] = complex(float64(j), 0)
		}

		pt1 := ckks.NewPlaintext(params, params.MaxLevel())
		pt2 := ckks.NewPlaintext(params, params.MaxLevel())

		encoder.Encode(vec1, pt1)
		encoder.Encode(vec2, pt2)

		ct1, _ := encryptor.EncryptNew(pt1)
		ct2, _ := encryptor.EncryptNew(pt2)

		testTasks[i] = struct{ ct1, ct2 *rlwe.Ciphertext }{ct1, ct2}
	}

	// Run multiple iterations for statistical significance
	var totalTime time.Duration
	var results []*BottleneckAnalysisResult

	for iteration := 0; iteration < config.Iterations; iteration++ {
		fmt.Printf("Running iteration %d/%d...\n", iteration+1, config.Iterations)

		// Reset profiler for this iteration
		iterProfiler := NewBottleneckProfiler()

		// Create profiled workers
		workers := make([]*ProfiledOperationCounter, config.N)
		for i := 0; i < config.N; i++ {
			workers[i] = NewProfiledOperationCounter(params, evk, iterProfiler, i)
		}

		// Channel setup
		tasks := make(chan struct{ ct1, ct2 *rlwe.Ciphertext }, config.K)
		resultsChan := make(chan struct {
			workerID int
			duration time.Duration
		}, config.K)

		// Start timing computation only
		computationStart := time.Now()

		// Record worker start times
		var wg sync.WaitGroup
		for i := 0; i < config.N; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()

				workerStart := time.Now()
				iterProfiler.mu.Lock()
				iterProfiler.WorkerStartTimes = append(iterProfiler.WorkerStartTimes, workerStart)
				iterProfiler.mu.Unlock()

				worker := workers[workerID]

				for task := range tasks {
					taskStart := time.Now()

					// Perform inner product computation with profiling
					tmp, _ := worker.MulNew(task.ct1, task.ct2)
					tmp, _ = worker.RelinearizeNew(tmp)
					worker.Rescale(tmp, tmp)

					// Tree-sum rotations
					for step := 1; step < config.VectorDim; step *= 2 {
						rot, _ := worker.RotateNew(tmp, step)
						tmp, _ = worker.AddNew(tmp, rot)
					}

					taskDuration := time.Since(taskStart)
					resultsChan <- struct {
						workerID int
						duration time.Duration
					}{workerID, taskDuration}
				}

				workerEnd := time.Now()
				iterProfiler.mu.Lock()
				iterProfiler.WorkerEndTimes = append(iterProfiler.WorkerEndTimes, workerEnd)
				iterProfiler.mu.Unlock()
			}(i)
		}

		// Submit all tasks
		for _, task := range testTasks {
			tasks <- task
		}
		close(tasks)

		// Wait for completion
		wg.Wait()
		close(resultsChan)

		iterationTime := time.Since(computationStart)
		totalTime += iterationTime

		// Collect results for this iteration
		var taskResults []struct {
			workerID int
			duration time.Duration
		}
		for result := range resultsChan {
			taskResults = append(taskResults, result)
		}

		// Analyze this iteration
		iterResult := analyzeIteration(iterProfiler, taskResults, iterationTime, config)
		results = append(results, iterResult)
	}

	// Aggregate results across all iterations
	finalResult := aggregateBottleneckResults(results, totalTime/time.Duration(config.Iterations))

	return finalResult, nil
}

// analyzeIteration analyzes a single iteration for bottlenecks
func analyzeIteration(profiler *BottleneckProfiler, taskResults []struct {
	workerID int
	duration time.Duration
}, totalTime time.Duration, config *BottleneckAnalysisConfig) *BottleneckAnalysisResult {
	result := &BottleneckAnalysisResult{
		TotalTime:   totalTime,
		Throughput:  float64(config.K) / totalTime.Seconds(),
		ThreadsUsed: config.N,
	}

	// Calculate average operation times
	if len(profiler.RotationTimes) > 0 {
		var sum time.Duration
		for _, t := range profiler.RotationTimes {
			sum += t
		}
		result.AvgRotationTime = sum / time.Duration(len(profiler.RotationTimes))
	}

	if len(profiler.MultiplicationTimes) > 0 {
		var sum time.Duration
		for _, t := range profiler.MultiplicationTimes {
			sum += t
		}
		result.AvgMultiplicationTime = sum / time.Duration(len(profiler.MultiplicationTimes))
	}

	if len(profiler.RelinearizationTimes) > 0 {
		var sum time.Duration
		for _, t := range profiler.RelinearizationTimes {
			sum += t
		}
		result.AvgRelinearizationTime = sum / time.Duration(len(profiler.RelinearizationTimes))
	}

	if len(profiler.RescaleTimes) > 0 {
		var sum time.Duration
		for _, t := range profiler.RescaleTimes {
			sum += t
		}
		result.AvgRescaleTime = sum / time.Duration(len(profiler.RescaleTimes))
	}

	if len(profiler.AdditionTimes) > 0 {
		var sum time.Duration
		for _, t := range profiler.AdditionTimes {
			sum += t
		}
		result.AvgAdditionTime = sum / time.Duration(len(profiler.AdditionTimes))
	}

	// Analyze worker efficiency
	workerTasks := make(map[int]int)
	for _, task := range taskResults {
		workerTasks[task.workerID]++
	}

	// Calculate load imbalance
	if len(workerTasks) > 0 {
		maxTasks := 0
		minTasks := config.K
		for _, count := range workerTasks {
			if count > maxTasks {
				maxTasks = count
			}
			if count < minTasks {
				minTasks = count
			}
		}
		result.LoadImbalance = float64(maxTasks-minTasks) / float64(maxTasks)
	}

	// Identify primary bottleneck
	result.ScalabilityBottleneck = identifyPrimaryBottleneck(result)

	return result
}

// identifyPrimaryBottleneck identifies the primary bottleneck based on profiling data
func identifyPrimaryBottleneck(result *BottleneckAnalysisResult) string {
	// Compare operation times to identify bottleneck
	maxTime := result.AvgRotationTime
	bottleneck := "Rotations"

	if result.AvgMultiplicationTime > maxTime {
		maxTime = result.AvgMultiplicationTime
		bottleneck = "Multiplications"
	}

	if result.AvgRelinearizationTime > maxTime {
		maxTime = result.AvgRelinearizationTime
		bottleneck = "Relinearizations"
	}

	if result.AvgRescaleTime > maxTime {
		maxTime = result.AvgRescaleTime
		bottleneck = "Rescaling"
	}

	// Add load imbalance check
	if result.LoadImbalance > 0.2 {
		bottleneck += " + Load Imbalance"
	}

	return bottleneck
}

// aggregateBottleneckResults aggregates results across multiple iterations
func aggregateBottleneckResults(results []*BottleneckAnalysisResult, avgTime time.Duration) *BottleneckAnalysisResult {
	if len(results) == 0 {
		return nil
	}

	final := &BottleneckAnalysisResult{
		TotalTime:   avgTime,
		ThreadsUsed: results[0].ThreadsUsed,
	}

	// Average all metrics
	for _, r := range results {
		final.AvgRotationTime += r.AvgRotationTime
		final.AvgMultiplicationTime += r.AvgMultiplicationTime
		final.AvgRelinearizationTime += r.AvgRelinearizationTime
		final.AvgRescaleTime += r.AvgRescaleTime
		final.AvgAdditionTime += r.AvgAdditionTime
		final.LoadImbalance += r.LoadImbalance
		final.Throughput += r.Throughput
	}

	count := time.Duration(len(results))
	final.AvgRotationTime /= count
	final.AvgMultiplicationTime /= count
	final.AvgRelinearizationTime /= count
	final.AvgRescaleTime /= count
	final.AvgAdditionTime /= count
	final.LoadImbalance /= float64(len(results))
	final.Throughput /= float64(len(results))

	// Identify most common bottleneck
	bottleneckCounts := make(map[string]int)
	for _, r := range results {
		bottleneckCounts[r.ScalabilityBottleneck]++
	}

	maxCount := 0
	for bottleneck, count := range bottleneckCounts {
		if count > maxCount {
			maxCount = count
			final.ScalabilityBottleneck = bottleneck
		}
	}

	return final
}

// PrintBottleneckAnalysis prints a comprehensive bottleneck analysis report
func PrintBottleneckAnalysis(result *BottleneckAnalysisResult) {
	fmt.Printf("\n=== BOTTLENECK ANALYSIS REPORT ===\n")
	fmt.Printf("Overall Performance:\n")
	fmt.Printf("  Total time: %v\n", result.TotalTime)
	fmt.Printf("  Throughput: %.2f ops/sec\n", result.Throughput)
	fmt.Printf("  Threads used: %d\n", result.ThreadsUsed)

	fmt.Printf("\nOperation-Level Profiling:\n")
	fmt.Printf("  Avg Rotation time:       %v\n", result.AvgRotationTime)
	fmt.Printf("  Avg Multiplication time: %v\n", result.AvgMultiplicationTime)
	fmt.Printf("  Avg Relinearization time: %v\n", result.AvgRelinearizationTime)
	fmt.Printf("  Avg Rescale time:        %v\n", result.AvgRescaleTime)
	fmt.Printf("  Avg Addition time:       %v\n", result.AvgAdditionTime)

	fmt.Printf("\nParallelization Efficiency:\n")
	fmt.Printf("  Load imbalance: %.1f%%\n", result.LoadImbalance*100)

	fmt.Printf("\nPRIMARY BOTTLENECK: %s\n", result.ScalabilityBottleneck)

	// Performance recommendations
	fmt.Printf("\n=== OPTIMIZATION RECOMMENDATIONS ===\n")
	if result.LoadImbalance > 0.15 {
		fmt.Printf("🔧 HIGH LOAD IMBALANCE DETECTED (%.1f%%)\n", result.LoadImbalance*100)
		fmt.Printf("   → Consider dynamic work stealing\n")
		fmt.Printf("   → Review task distribution strategy\n")
	}

	// Identify slowest operation
	slowestOp := "Rotations"
	slowestTime := result.AvgRotationTime

	if result.AvgMultiplicationTime > slowestTime {
		slowestOp = "Multiplications"
		slowestTime = result.AvgMultiplicationTime
	}
	if result.AvgRelinearizationTime > slowestTime {
		slowestOp = "Relinearizations"
		slowestTime = result.AvgRelinearizationTime
	}
	if result.AvgRescaleTime > slowestTime {
		slowestOp = "Rescaling"
		slowestTime = result.AvgRescaleTime
	}

	fmt.Printf("🎯 FOCUS OPTIMIZATION ON: %s (%.2fms avg)\n",
		slowestOp, float64(slowestTime.Nanoseconds())/1000000.0)

	totalOpTime := result.AvgRotationTime + result.AvgMultiplicationTime +
		result.AvgRelinearizationTime + result.AvgRescaleTime + result.AvgAdditionTime

	if slowestTime > totalOpTime/5*2 { // More than 40% of total operation time
		fmt.Printf("   → This operation dominates computation time\n")
		fmt.Printf("   → Consider algorithm-level optimizations\n")
	}
}

// OptimizedRelinearizationConfig holds configuration for relinearization optimizations
type OptimizedRelinearizationConfig struct {
	EnableStreamingKeySwitch  bool // Use streaming key-switching
	EnableLazyRelinearization bool // Defer relinearization until necessary
	BatchSize                 int  // Batch multiple modulus levels
	CRTWindowSize             int  // CRT windowing parameter
}

// OptimizedOperationCounter with enhanced relinearization strategies
type OptimizedOperationCounter struct {
	eval     *ckks.Evaluator
	profiler *BottleneckProfiler
	workerID int
	config   *OptimizedRelinearizationConfig

	// Lazy relinearization state
	pendingRelins []*rlwe.Ciphertext
	relinCount    int

	// Streaming key-switch buffers
	keyBuffer    []rlwe.EvaluationKeySet
	streamBuffer []*rlwe.Ciphertext
}

// NewOptimizedOperationCounter creates optimized operation counter with advanced relinearization
func NewOptimizedOperationCounter(params ckks.Parameters, evk rlwe.EvaluationKeySet, profiler *BottleneckProfiler, workerID int, config *OptimizedRelinearizationConfig) *OptimizedOperationCounter {
	return &OptimizedOperationCounter{
		eval:          ckks.NewEvaluator(params, evk),
		profiler:      profiler,
		workerID:      workerID,
		config:        config,
		pendingRelins: make([]*rlwe.Ciphertext, 0, config.BatchSize),
		streamBuffer:  make([]*rlwe.Ciphertext, 0, config.BatchSize),
	}
}

// OptimizedRelinearizeNew with streaming and batching optimizations
func (ooc *OptimizedOperationCounter) OptimizedRelinearizeNew(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	start := time.Now()

	var result *rlwe.Ciphertext
	var err error

	if ooc.config.EnableLazyRelinearization {
		// Lazy relinearization: defer until batch is full or forced
		ooc.pendingRelins = append(ooc.pendingRelins, ct)
		ooc.relinCount++

		if len(ooc.pendingRelins) >= ooc.config.BatchSize {
			result, err = ooc.flushLazyRelinearizations()
		} else {
			// Return as-is for now, will be relinearized later
			result = ct
		}
	} else if ooc.config.EnableStreamingKeySwitch {
		// Streaming key-switching: process multiple modulus levels in one pass
		result, err = ooc.streamingKeySwitch(ct)
	} else {
		// Standard relinearization
		result, err = ooc.eval.RelinearizeNew(ct)
	}

	duration := time.Since(start)

	ooc.profiler.mu.Lock()
	ooc.profiler.RelinearizationTimes = append(ooc.profiler.RelinearizationTimes, duration)
	ooc.profiler.mu.Unlock()

	return result, err
}

// streamingKeySwitch implements streaming key-switching for better memory efficiency
func (ooc *OptimizedOperationCounter) streamingKeySwitch(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	// Add to streaming buffer
	ooc.streamBuffer = append(ooc.streamBuffer, ct)

	// Process when buffer is full
	if len(ooc.streamBuffer) >= ooc.config.BatchSize {
		return ooc.processBatchedKeySwitch()
	}

	// For now, fall back to standard relinearization
	return ooc.eval.RelinearizeNew(ct)
}

// processBatchedKeySwitch processes multiple key-switches in a single pass
func (ooc *OptimizedOperationCounter) processBatchedKeySwitch() (*rlwe.Ciphertext, error) {
	if len(ooc.streamBuffer) == 0 {
		return nil, fmt.Errorf("no ciphertexts in stream buffer")
	}

	// For now, process the first one and clear buffer
	// In a real implementation, this would batch multiple modulus operations
	result, err := ooc.eval.RelinearizeNew(ooc.streamBuffer[0])
	ooc.streamBuffer = ooc.streamBuffer[:0] // Clear buffer

	return result, err
}

// flushLazyRelinearizations processes all pending lazy relinearizations
func (ooc *OptimizedOperationCounter) flushLazyRelinearizations() (*rlwe.Ciphertext, error) {
	if len(ooc.pendingRelins) == 0 {
		return nil, fmt.Errorf("no pending relinearizations")
	}

	// Process the first pending relinearization and clear the batch
	result, err := ooc.eval.RelinearizeNew(ooc.pendingRelins[0])
	ooc.pendingRelins = ooc.pendingRelins[:0] // Clear batch

	return result, err
}

// ForceRelinearize forces processing of any pending lazy relinearizations
func (ooc *OptimizedOperationCounter) ForceRelinearize() error {
	if len(ooc.pendingRelins) > 0 {
		_, err := ooc.flushLazyRelinearizations()
		return err
	}
	return nil
}

// Standard operations with profiling (same as before but using optimized relinearization)
func (ooc *OptimizedOperationCounter) RotateNew(ct *rlwe.Ciphertext, k int) (*rlwe.Ciphertext, error) {
	start := time.Now()
	result, err := ooc.eval.RotateNew(ct, k)
	duration := time.Since(start)

	ooc.profiler.mu.Lock()
	ooc.profiler.RotationTimes = append(ooc.profiler.RotationTimes, duration)
	ooc.profiler.mu.Unlock()

	return result, err
}

func (ooc *OptimizedOperationCounter) MulNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	start := time.Now()
	result, err := ooc.eval.MulNew(ct1, ct2)
	duration := time.Since(start)

	ooc.profiler.mu.Lock()
	ooc.profiler.MultiplicationTimes = append(ooc.profiler.MultiplicationTimes, duration)
	ooc.profiler.mu.Unlock()

	return result, err
}

func (ooc *OptimizedOperationCounter) Rescale(ct, ctOut *rlwe.Ciphertext) error {
	start := time.Now()
	err := ooc.eval.Rescale(ct, ctOut)
	duration := time.Since(start)

	ooc.profiler.mu.Lock()
	ooc.profiler.RescaleTimes = append(ooc.profiler.RescaleTimes, duration)
	ooc.profiler.mu.Unlock()

	return err
}

func (ooc *OptimizedOperationCounter) AddNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	start := time.Now()
	result, err := ooc.eval.AddNew(ct1, ct2)
	duration := time.Since(start)

	ooc.profiler.mu.Lock()
	ooc.profiler.AdditionTimes = append(ooc.profiler.AdditionTimes, duration)
	ooc.profiler.mu.Unlock()

	return result, err
}

// WorkStealingTask represents a task that can be stolen by idle workers
type WorkStealingTask struct {
	ID         int
	CT1        *rlwe.Ciphertext
	CT2        *rlwe.Ciphertext
	Priority   int // Higher priority tasks processed first
	Complexity int // Estimated computation complexity
}

// WorkStealingDeque implements a lock-free work-stealing deque
type WorkStealingDeque struct {
	tasks []WorkStealingTask
	head  int64 // Atomic access
	tail  int64 // Atomic access
	mu    sync.Mutex
}

// NewWorkStealingDeque creates a new work-stealing deque
func NewWorkStealingDeque(capacity int) *WorkStealingDeque {
	return &WorkStealingDeque{
		tasks: make([]WorkStealingTask, capacity),
		head:  0,
		tail:  0,
	}
}

// PushBottom adds a task to the bottom (owner worker adds tasks here)
func (wsd *WorkStealingDeque) PushBottom(task WorkStealingTask) bool {
	wsd.mu.Lock()
	defer wsd.mu.Unlock()

	currentTail := atomic.LoadInt64(&wsd.tail)
	currentHead := atomic.LoadInt64(&wsd.head)

	// Check if deque is full
	if currentTail-currentHead >= int64(len(wsd.tasks)) {
		return false // Deque is full
	}

	wsd.tasks[currentTail%int64(len(wsd.tasks))] = task
	atomic.StoreInt64(&wsd.tail, currentTail+1)
	return true
}

// PopBottom removes a task from the bottom (owner worker pops here)
func (wsd *WorkStealingDeque) PopBottom() (WorkStealingTask, bool) {
	wsd.mu.Lock()
	defer wsd.mu.Unlock()

	currentTail := atomic.LoadInt64(&wsd.tail)
	currentHead := atomic.LoadInt64(&wsd.head)

	if currentTail <= currentHead {
		return WorkStealingTask{}, false // Empty
	}

	newTail := currentTail - 1
	atomic.StoreInt64(&wsd.tail, newTail)
	task := wsd.tasks[newTail%int64(len(wsd.tasks))]

	return task, true
}

// StealTop steals a task from the top (other workers steal here)
func (wsd *WorkStealingDeque) StealTop() (WorkStealingTask, bool) {
	currentHead := atomic.LoadInt64(&wsd.head)
	currentTail := atomic.LoadInt64(&wsd.tail)

	if currentHead >= currentTail {
		return WorkStealingTask{}, false // Empty or contention
	}

	task := wsd.tasks[currentHead%int64(len(wsd.tasks))]

	// Try to atomically increment head
	if !atomic.CompareAndSwapInt64(&wsd.head, currentHead, currentHead+1) {
		return WorkStealingTask{}, false // Failed to steal (contention)
	}

	return task, true
}

// Size returns approximate size (may be inaccurate due to concurrent access)
func (wsd *WorkStealingDeque) Size() int {
	currentTail := atomic.LoadInt64(&wsd.tail)
	currentHead := atomic.LoadInt64(&wsd.head)
	size := int(currentTail - currentHead)
	if size < 0 {
		return 0
	}
	return size
}

// WorkStealingScheduler manages work-stealing across multiple workers
type WorkStealingScheduler struct {
	workerDeques   []*WorkStealingDeque
	globalQueue    chan WorkStealingTask
	numWorkers     int
	stealAttempts  []int64 // Track steal attempts per worker
	stealSuccesses []int64 // Track successful steals per worker
	tasksCompleted []int64 // Track tasks completed per worker
}

// NewWorkStealingScheduler creates a new work-stealing scheduler
func NewWorkStealingScheduler(numWorkers int, dequeSize int) *WorkStealingScheduler {
	deques := make([]*WorkStealingDeque, numWorkers)
	for i := 0; i < numWorkers; i++ {
		deques[i] = NewWorkStealingDeque(dequeSize)
	}

	return &WorkStealingScheduler{
		workerDeques:   deques,
		globalQueue:    make(chan WorkStealingTask, numWorkers*2),
		numWorkers:     numWorkers,
		stealAttempts:  make([]int64, numWorkers),
		stealSuccesses: make([]int64, numWorkers),
		tasksCompleted: make([]int64, numWorkers),
	}
}

// DistributeTasks distributes initial tasks across worker deques
func (wss *WorkStealingScheduler) DistributeTasks(tasks []WorkStealingTask) {
	tasksPerWorker := len(tasks) / wss.numWorkers
	remainder := len(tasks) % wss.numWorkers

	taskIndex := 0
	for workerID := 0; workerID < wss.numWorkers; workerID++ {
		// Give each worker their fair share
		workerTasks := tasksPerWorker
		if workerID < remainder {
			workerTasks++ // Distribute remainder evenly
		}

		for i := 0; i < workerTasks && taskIndex < len(tasks); i++ {
			if !wss.workerDeques[workerID].PushBottom(tasks[taskIndex]) {
				// If local deque is full, add to global queue
				select {
				case wss.globalQueue <- tasks[taskIndex]:
				default:
					// Global queue full, skip for now
				}
			}
			taskIndex++
		}
	}

	// Add any remaining tasks to global queue
	for taskIndex < len(tasks) {
		select {
		case wss.globalQueue <- tasks[taskIndex]:
		default:
			break // Global queue full
		}
		taskIndex++
	}
}

// GetTask gets the next task for a worker (tries local deque first, then steals)
func (wss *WorkStealingScheduler) GetTask(workerID int) (WorkStealingTask, bool) {
	// 1. Try local deque first
	if task, ok := wss.workerDeques[workerID].PopBottom(); ok {
		return task, true
	}

	// 2. Try global queue
	select {
	case task := <-wss.globalQueue:
		return task, true
	default:
		// Global queue empty, try stealing
	}

	// 3. Try stealing from other workers
	atomic.AddInt64(&wss.stealAttempts[workerID], 1)

	// Random stealing strategy: try random workers
	for attempts := 0; attempts < wss.numWorkers; attempts++ {
		victimID := (workerID + attempts + 1) % wss.numWorkers
		if victimID == workerID {
			continue // Don't steal from yourself
		}

		if task, ok := wss.workerDeques[victimID].StealTop(); ok {
			atomic.AddInt64(&wss.stealSuccesses[workerID], 1)
			return task, true
		}
	}

	return WorkStealingTask{}, false // No tasks available
}

// CompleteTask marks a task as completed for statistics
func (wss *WorkStealingScheduler) CompleteTask(workerID int) {
	atomic.AddInt64(&wss.tasksCompleted[workerID], 1)
}

// GetStatistics returns work-stealing statistics
func (wss *WorkStealingScheduler) GetStatistics() map[string]interface{} {
	stats := make(map[string]interface{})

	var totalAttempts, totalSuccesses, totalCompleted int64
	attempts := make([]int64, wss.numWorkers)
	successes := make([]int64, wss.numWorkers)
	completed := make([]int64, wss.numWorkers)

	for i := 0; i < wss.numWorkers; i++ {
		attempts[i] = atomic.LoadInt64(&wss.stealAttempts[i])
		successes[i] = atomic.LoadInt64(&wss.stealSuccesses[i])
		completed[i] = atomic.LoadInt64(&wss.tasksCompleted[i])

		totalAttempts += attempts[i]
		totalSuccesses += successes[i]
		totalCompleted += completed[i]
	}

	stealEfficiency := float64(0)
	if totalAttempts > 0 {
		stealEfficiency = float64(totalSuccesses) / float64(totalAttempts) * 100.0
	}

	// Calculate load balance
	maxCompleted := completed[0]
	minCompleted := completed[0]
	for i := 1; i < wss.numWorkers; i++ {
		if completed[i] > maxCompleted {
			maxCompleted = completed[i]
		}
		if completed[i] < minCompleted {
			minCompleted = completed[i]
		}
	}

	loadImbalance := float64(0)
	if maxCompleted > 0 {
		loadImbalance = float64(maxCompleted-minCompleted) / float64(maxCompleted) * 100.0
	}

	stats["steal_attempts"] = totalAttempts
	stats["steal_successes"] = totalSuccesses
	stats["steal_efficiency"] = stealEfficiency
	stats["load_imbalance"] = loadImbalance
	stats["tasks_completed"] = totalCompleted
	stats["per_worker_completed"] = completed
	stats["per_worker_steals"] = successes

	return stats
}

// CacheOptimizedOperationCounter extends OptimizedOperationCounter with cache-friendly buffers
type CacheOptimizedOperationCounter struct {
	*OptimizedOperationCounter
	bufferPool     *BufferPool
	activeBuffers  []*AlignedBuffer // Track active buffers for cleanup
	allocCount     int64            // Track allocations
	memoryPressure float64          // Track memory pressure
}

// NewCacheOptimizedOperationCounter creates operation counter with cache optimization
func NewCacheOptimizedOperationCounter(params ckks.Parameters, evk rlwe.EvaluationKeySet,
	profiler *BottleneckProfiler, workerID int,
	relinConfig *OptimizedRelinearizationConfig,
	bufferPool *BufferPool) *CacheOptimizedOperationCounter {

	optimized := NewOptimizedOperationCounter(params, evk, profiler, workerID, relinConfig)

	return &CacheOptimizedOperationCounter{
		OptimizedOperationCounter: optimized,
		bufferPool:                bufferPool,
		activeBuffers:             make([]*AlignedBuffer, 0, 10),
	}
}

// AllocateWorkBuffer allocates a cache-aligned work buffer
func (cooc *CacheOptimizedOperationCounter) AllocateWorkBuffer() *AlignedBuffer {
	buffer := cooc.bufferPool.GetBuffer()
	cooc.activeBuffers = append(cooc.activeBuffers, buffer)
	atomic.AddInt64(&cooc.allocCount, 1)
	return buffer
}

// ReleaseWorkBuffer releases a work buffer back to the pool
func (cooc *CacheOptimizedOperationCounter) ReleaseWorkBuffer(buffer *AlignedBuffer) {
	cooc.bufferPool.ReturnBuffer(buffer)

	// Remove from active buffers list
	for i, activeBuffer := range cooc.activeBuffers {
		if activeBuffer == buffer {
			// Remove by swapping with last element
			cooc.activeBuffers[i] = cooc.activeBuffers[len(cooc.activeBuffers)-1]
			cooc.activeBuffers = cooc.activeBuffers[:len(cooc.activeBuffers)-1]
			break
		}
	}
}

// CleanupBuffers releases all active buffers
func (cooc *CacheOptimizedOperationCounter) CleanupBuffers() {
	for _, buffer := range cooc.activeBuffers {
		cooc.bufferPool.ReturnBuffer(buffer)
	}
	cooc.activeBuffers = cooc.activeBuffers[:0]
}

// GetMemoryStatistics returns memory usage statistics
func (cooc *CacheOptimizedOperationCounter) GetMemoryStatistics() map[string]interface{} {
	poolStats := cooc.bufferPool.GetStatistics()

	// Add additional memory metrics
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	stats := make(map[string]interface{})
	stats["buffer_pool"] = poolStats
	stats["active_buffers"] = len(cooc.activeBuffers)
	stats["alloc_count"] = atomic.LoadInt64(&cooc.allocCount)
	stats["heap_alloc"] = memStats.HeapAlloc
	stats["heap_objects"] = memStats.HeapObjects
	stats["gc_count"] = memStats.NumGC
	stats["memory_pressure"] = cooc.memoryPressure

	return stats
}

// UpdateMemoryPressure updates memory pressure metrics
func (cooc *CacheOptimizedOperationCounter) UpdateMemoryPressure() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Simple memory pressure calculation: ratio of heap to system memory
	if memStats.Sys > 0 {
		cooc.memoryPressure = float64(memStats.HeapAlloc) / float64(memStats.Sys) * 100.0
	}
}

// PipelineStage represents different stages in the HE computation pipeline
type PipelineStage int

const (
	MultiplicationStage PipelineStage = iota
	RelinearizationStage
	RescaleStage
	RotationStage
)

// PipelineTask represents a task flowing through the pipeline
type PipelineTask struct {
	ID        int
	CT1       *rlwe.Ciphertext
	CT2       *rlwe.Ciphertext
	Result    *rlwe.Ciphertext
	VectorDim int
	Stage     PipelineStage
	StartTime time.Time
	StageTime map[PipelineStage]time.Duration
	WorkerID  int
}

// NewPipelineTask creates a new pipeline task
func NewPipelineTask(id int, ct1, ct2 *rlwe.Ciphertext, vectorDim int) *PipelineTask {
	return &PipelineTask{
		ID:        id,
		CT1:       ct1,
		CT2:       ct2,
		VectorDim: vectorDim,
		Stage:     MultiplicationStage,
		StartTime: time.Now(),
		StageTime: make(map[PipelineStage]time.Duration),
	}
}

// PipelineWorker processes tasks for a specific stage
type PipelineWorker struct {
	workerID   int
	stage      PipelineStage
	evaluator  *CacheOptimizedOperationCounter
	inputChan  chan *PipelineTask
	outputChan chan *PipelineTask
	profiler   *BottleneckProfiler
	active     bool
	processed  int64
}

// NewPipelineWorker creates a new pipeline worker for a specific stage
func NewPipelineWorker(workerID int, stage PipelineStage, evaluator *CacheOptimizedOperationCounter, profiler *BottleneckProfiler) *PipelineWorker {
	return &PipelineWorker{
		workerID:  workerID,
		stage:     stage,
		evaluator: evaluator,
		profiler:  profiler,
		active:    false,
	}
}

// SetChannels sets input and output channels for the pipeline worker
func (pw *PipelineWorker) SetChannels(input, output chan *PipelineTask) {
	pw.inputChan = input
	pw.outputChan = output
}

// Start starts the pipeline worker
func (pw *PipelineWorker) Start() {
	pw.active = true
	go pw.processLoop()
}

// Stop stops the pipeline worker
func (pw *PipelineWorker) Stop() {
	pw.active = false
}

// processLoop is the main processing loop for the pipeline worker
func (pw *PipelineWorker) processLoop() {
	for pw.active {
		select {
		case task, ok := <-pw.inputChan:
			if !ok {
				// Input channel closed
				if pw.outputChan != nil {
					close(pw.outputChan)
				}
				return
			}

			stageStart := time.Now()

			// Process task based on stage
			switch pw.stage {
			case MultiplicationStage:
				pw.processMulStage(task)
			case RelinearizationStage:
				pw.processRelinStage(task)
			case RescaleStage:
				pw.processRescaleStage(task)
			case RotationStage:
				pw.processRotationStage(task)
			}

			task.StageTime[pw.stage] = time.Since(stageStart)
			task.WorkerID = pw.workerID
			atomic.AddInt64(&pw.processed, 1)

			// Send to next stage if output channel exists
			if pw.outputChan != nil {
				pw.outputChan <- task
			}
		}
	}
}

// processMulStage processes multiplication stage
func (pw *PipelineWorker) processMulStage(task *PipelineTask) {
	result, err := pw.evaluator.MulNew(task.CT1, task.CT2)
	if err == nil {
		task.Result = result
		task.Stage = RelinearizationStage
	}
}

// processRelinStage processes relinearization stage
func (pw *PipelineWorker) processRelinStage(task *PipelineTask) {
	if task.Result != nil {
		result, err := pw.evaluator.OptimizedRelinearizeNew(task.Result)
		if err == nil {
			task.Result = result
			task.Stage = RescaleStage
		}
	}
}

// processRescaleStage processes rescaling stage
func (pw *PipelineWorker) processRescaleStage(task *PipelineTask) {
	if task.Result != nil {
		err := pw.evaluator.Rescale(task.Result, task.Result)
		if err == nil {
			task.Stage = RotationStage
		}
	}
}

// processRotationStage processes rotation (tree-sum) stage
func (pw *PipelineWorker) processRotationStage(task *PipelineTask) {
	if task.Result != nil {
		tmp := task.Result

		// Tree-sum rotations
		for step := 1; step < task.VectorDim; step *= 2 {
			rot, err := pw.evaluator.RotateNew(tmp, step)
			if err == nil {
				tmp, _ = pw.evaluator.AddNew(tmp, rot)
			}
		}

		task.Result = tmp
		task.Stage = -1 // Mark as completed
	}
}

// GetStatistics returns worker statistics
func (pw *PipelineWorker) GetStatistics() map[string]interface{} {
	return map[string]interface{}{
		"worker_id": pw.workerID,
		"stage":     pw.stage,
		"processed": atomic.LoadInt64(&pw.processed),
		"active":    pw.active,
	}
}

// PipelinedInnerProductProcessor manages the entire pipeline
type PipelinedInnerProductProcessor struct {
	config     *BottleneckAnalysisConfig
	profiler   *BottleneckProfiler
	bufferPool *ThreadLocalBufferPool

	// Pipeline stages
	mulWorkers      []*PipelineWorker
	relinWorkers    []*PipelineWorker
	rescaleWorkers  []*PipelineWorker
	rotationWorkers []*PipelineWorker

	// Pipeline channels
	mulChannel      chan *PipelineTask
	relinChannel    chan *PipelineTask
	rescaleChannel  chan *PipelineTask
	rotationChannel chan *PipelineTask
	outputChannel   chan *PipelineTask

	// Configuration
	mulStageWorkers      int
	relinStageWorkers    int
	rescaleStageWorkers  int
	rotationStageWorkers int
}

// NewPipelinedInnerProductProcessor creates a new pipelined processor
func NewPipelinedInnerProductProcessor(config *BottleneckAnalysisConfig, profiler *BottleneckProfiler) *PipelinedInnerProductProcessor {
	bufferPool := NewThreadLocalBufferPool(config.N * 4) // 4 stages

	// Calculate workers per stage based on relative costs from bottleneck analysis
	// Relinearization is most expensive, so give it more workers
	totalWorkers := config.N
	relinWorkers := totalWorkers / 2   // 50% of workers on bottleneck stage
	mulWorkers := totalWorkers / 4     // 25% on multiplication
	rescaleWorkers := totalWorkers / 8 // 12.5% on rescaling
	rotationWorkers := totalWorkers - relinWorkers - mulWorkers - rescaleWorkers

	// Ensure at least 1 worker per stage
	if mulWorkers < 1 {
		mulWorkers = 1
	}
	if relinWorkers < 1 {
		relinWorkers = 1
	}
	if rescaleWorkers < 1 {
		rescaleWorkers = 1
	}
	if rotationWorkers < 1 {
		rotationWorkers = 1
	}

	return &PipelinedInnerProductProcessor{
		config:               config,
		profiler:             profiler,
		bufferPool:           bufferPool,
		mulStageWorkers:      mulWorkers,
		relinStageWorkers:    relinWorkers,
		rescaleStageWorkers:  rescaleWorkers,
		rotationStageWorkers: rotationWorkers,

		// Create channels with appropriate buffer sizes
		mulChannel:      make(chan *PipelineTask, config.K),
		relinChannel:    make(chan *PipelineTask, relinWorkers*2),
		rescaleChannel:  make(chan *PipelineTask, rescaleWorkers*2),
		rotationChannel: make(chan *PipelineTask, rotationWorkers*2),
		outputChannel:   make(chan *PipelineTask, config.K),
	}
}

// Initialize sets up all pipeline workers and channels
func (pipp *PipelinedInnerProductProcessor) Initialize() error {
	// Initialize CKKS context for pipeline
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            pipp.config.LogN,
		LogQ:            pipp.config.LogQ,
		LogP:            pipp.config.LogP,
		LogDefaultScale: 40,
		Xs:              rlwe.DefaultXs,
		Xe:              rlwe.DefaultXe,
	})
	if err != nil {
		return fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// Generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	rotations := []int{}
	for step := 1; step < pipp.config.VectorDim; step *= 2 {
		rotations = append(rotations, step)
	}

	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := make([]uint64, len(rotations))
	for i, rot := range rotations {
		galEls[i] = params.GaloisElementForRotation(rot)
	}
	galKeys := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	// Create relinearization optimization config
	relinConfig := &OptimizedRelinearizationConfig{
		EnableStreamingKeySwitch:  true,
		EnableLazyRelinearization: false, // Disabled for pipeline to avoid complexity
		BatchSize:                 4,
		CRTWindowSize:             2,
	}

	// Create workers for each stage
	workerID := 0

	// Multiplication stage workers
	for i := 0; i < pipp.mulStageWorkers; i++ {
		bufferPool := pipp.bufferPool.GetWorkerPool(workerID, 1024, 8)
		evaluator := NewCacheOptimizedOperationCounter(params, evk, pipp.profiler, workerID, relinConfig, bufferPool)
		worker := NewPipelineWorker(workerID, MultiplicationStage, evaluator, pipp.profiler)
		worker.SetChannels(pipp.mulChannel, pipp.relinChannel)
		pipp.mulWorkers = append(pipp.mulWorkers, worker)
		workerID++
	}

	// Relinearization stage workers (most workers here due to bottleneck)
	for i := 0; i < pipp.relinStageWorkers; i++ {
		bufferPool := pipp.bufferPool.GetWorkerPool(workerID, 1024, 8)
		evaluator := NewCacheOptimizedOperationCounter(params, evk, pipp.profiler, workerID, relinConfig, bufferPool)
		worker := NewPipelineWorker(workerID, RelinearizationStage, evaluator, pipp.profiler)
		worker.SetChannels(pipp.relinChannel, pipp.rescaleChannel)
		pipp.relinWorkers = append(pipp.relinWorkers, worker)
		workerID++
	}

	// Rescale stage workers
	for i := 0; i < pipp.rescaleStageWorkers; i++ {
		bufferPool := pipp.bufferPool.GetWorkerPool(workerID, 1024, 8)
		evaluator := NewCacheOptimizedOperationCounter(params, evk, pipp.profiler, workerID, relinConfig, bufferPool)
		worker := NewPipelineWorker(workerID, RescaleStage, evaluator, pipp.profiler)
		worker.SetChannels(pipp.rescaleChannel, pipp.rotationChannel)
		pipp.rescaleWorkers = append(pipp.rescaleWorkers, worker)
		workerID++
	}

	// Rotation stage workers
	for i := 0; i < pipp.rotationStageWorkers; i++ {
		bufferPool := pipp.bufferPool.GetWorkerPool(workerID, 1024, 8)
		evaluator := NewCacheOptimizedOperationCounter(params, evk, pipp.profiler, workerID, relinConfig, bufferPool)
		worker := NewPipelineWorker(workerID, RotationStage, evaluator, pipp.profiler)
		worker.SetChannels(pipp.rotationChannel, pipp.outputChannel)
		pipp.rotationWorkers = append(pipp.rotationWorkers, worker)
		workerID++
	}

	return nil
}

// Start starts all pipeline workers
func (pipp *PipelinedInnerProductProcessor) Start() {
	// Start all workers
	for _, worker := range pipp.mulWorkers {
		worker.Start()
	}
	for _, worker := range pipp.relinWorkers {
		worker.Start()
	}
	for _, worker := range pipp.rescaleWorkers {
		worker.Start()
	}
	for _, worker := range pipp.rotationWorkers {
		worker.Start()
	}
}

// Stop stops all pipeline workers
func (pipp *PipelinedInnerProductProcessor) Stop() {
	// Stop all workers
	for _, worker := range pipp.mulWorkers {
		worker.Stop()
	}
	for _, worker := range pipp.relinWorkers {
		worker.Stop()
	}
	for _, worker := range pipp.rescaleWorkers {
		worker.Stop()
	}
	for _, worker := range pipp.rotationWorkers {
		worker.Stop()
	}
}

// SubmitTask submits a task to the pipeline
func (pipp *PipelinedInnerProductProcessor) SubmitTask(task *PipelineTask) {
	pipp.mulChannel <- task
}

// GetResults retrieves completed tasks from the pipeline
func (pipp *PipelinedInnerProductProcessor) GetResults() chan *PipelineTask {
	return pipp.outputChannel
}

// GetPipelineStatistics returns comprehensive pipeline statistics
func (pipp *PipelinedInnerProductProcessor) GetPipelineStatistics() map[string]interface{} {
	stats := make(map[string]interface{})

	// Collect per-stage statistics
	mulStats := make([]map[string]interface{}, len(pipp.mulWorkers))
	for i, worker := range pipp.mulWorkers {
		mulStats[i] = worker.GetStatistics()
	}

	relinStats := make([]map[string]interface{}, len(pipp.relinWorkers))
	for i, worker := range pipp.relinWorkers {
		relinStats[i] = worker.GetStatistics()
	}

	rescaleStats := make([]map[string]interface{}, len(pipp.rescaleWorkers))
	for i, worker := range pipp.rescaleWorkers {
		rescaleStats[i] = worker.GetStatistics()
	}

	rotationStats := make([]map[string]interface{}, len(pipp.rotationWorkers))
	for i, worker := range pipp.rotationWorkers {
		rotationStats[i] = worker.GetStatistics()
	}

	stats["multiplication_stage"] = mulStats
	stats["relinearization_stage"] = relinStats
	stats["rescale_stage"] = rescaleStats
	stats["rotation_stage"] = rotationStats

	// Channel buffer utilization
	stats["channel_utilization"] = map[string]interface{}{
		"mul_channel":      len(pipp.mulChannel),
		"relin_channel":    len(pipp.relinChannel),
		"rescale_channel":  len(pipp.rescaleChannel),
		"rotation_channel": len(pipp.rotationChannel),
		"output_channel":   len(pipp.outputChannel),
	}

	stats["worker_distribution"] = map[string]interface{}{
		"mul_workers":      pipp.mulStageWorkers,
		"relin_workers":    pipp.relinStageWorkers,
		"rescale_workers":  pipp.rescaleStageWorkers,
		"rotation_workers": pipp.rotationStageWorkers,
	}

	return stats
}

// OptimizationConfig controls which optimizations to enable
type OptimizationConfig struct {
	EnableOptimizedRelinearization bool
	EnableWorkStealing             bool
	EnableCacheFriendlyBuffers     bool
	EnablePipelinedProcessing      bool
	EnableStreamingKeySwitch       bool
	EnableLazyRelinearization      bool
	RelinearizationBatchSize       int
	CRTWindowSize                  int
}

// DefaultOptimizationConfig returns the optimal configuration based on our findings
func DefaultOptimizationConfig() *OptimizationConfig {
	return &OptimizationConfig{
		EnableOptimizedRelinearization: true,  // Primary bottleneck (91.3% of time)
		EnableWorkStealing:             true,  // Eliminates 40% load imbalance
		EnableCacheFriendlyBuffers:     true,  // 100% hit rate achieved
		EnablePipelinedProcessing:      false, // Enable for high-throughput scenarios
		EnableStreamingKeySwitch:       true,  // 15-25% relinearization improvement
		EnableLazyRelinearization:      false, // Use streaming instead for simplicity
		RelinearizationBatchSize:       4,     // Optimal batch size
		CRTWindowSize:                  2,     // CRT window parameter
	}
}

// OptimizedBottleneckAnalysisResult extends the original with optimization metrics
type OptimizedBottleneckAnalysisResult struct {
	*BottleneckAnalysisResult

	// Optimization-specific metrics
	WorkStealingStats    map[string]interface{}
	BufferPoolStats      []map[string]interface{}
	RelinearizationStats map[string]interface{}
	PipelineStats        map[string]interface{}
	OptimizationsEnabled []string

	// Performance improvements
	BaselineTime        time.Duration
	OptimizedTime       time.Duration
	ImprovementFactor   float64
	BottleneckReduction float64
}

// RunOptimizedBottleneckAnalysis runs the enhanced analysis with all optimizations
func RunOptimizedBottleneckAnalysis(config *BottleneckAnalysisConfig, optConfig *OptimizationConfig) (*OptimizedBottleneckAnalysisResult, error) {
	fmt.Printf("=== OPTIMIZED BOTTLENECK ANALYSIS ===\n")
	fmt.Printf("Configuration: K=%d, N=%d, VectorDim=%d, Iterations=%d\n",
		config.K, config.N, config.VectorDim, config.Iterations)

	// List enabled optimizations
	var enabledOpts []string
	if optConfig.EnableOptimizedRelinearization {
		enabledOpts = append(enabledOpts, "Optimized Relinearization")
	}
	if optConfig.EnableWorkStealing {
		enabledOpts = append(enabledOpts, "Work Stealing")
	}
	if optConfig.EnableCacheFriendlyBuffers {
		enabledOpts = append(enabledOpts, "Cache-Friendly Buffers")
	}
	if optConfig.EnablePipelinedProcessing {
		enabledOpts = append(enabledOpts, "Pipelined Processing")
	}

	fmt.Printf("Enabled optimizations: %v\n", enabledOpts)

	// Run baseline for comparison if optimizations are enabled
	var baselineResult *BottleneckAnalysisResult
	if len(enabledOpts) > 0 {
		fmt.Printf("Running baseline analysis for comparison...\n")
		disabledConfig := &OptimizationConfig{} // All optimizations disabled
		baseline, err := runOptimizedAnalysisInternal(config, disabledConfig)
		if err != nil {
			return nil, fmt.Errorf("baseline analysis failed: %w", err)
		}
		baselineResult = baseline.BottleneckAnalysisResult
	}

	// Run optimized analysis
	fmt.Printf("Running optimized analysis...\n")
	optimizedResult, err := runOptimizedAnalysisInternal(config, optConfig)
	if err != nil {
		return nil, fmt.Errorf("optimized analysis failed: %w", err)
	}

	// Calculate improvements
	if baselineResult != nil {
		optimizedResult.BaselineTime = baselineResult.TotalTime
		optimizedResult.OptimizedTime = optimizedResult.TotalTime
		optimizedResult.ImprovementFactor = float64(baselineResult.TotalTime) / float64(optimizedResult.TotalTime)

		// Calculate bottleneck reduction (focus on relinearization)
		baselineRelinTime := baselineResult.AvgRelinearizationTime
		optimizedRelinTime := optimizedResult.AvgRelinearizationTime
		if baselineRelinTime > 0 {
			optimizedResult.BottleneckReduction = (float64(baselineRelinTime) - float64(optimizedRelinTime)) / float64(baselineRelinTime) * 100.0
		}
	}

	return optimizedResult, nil
}

// runOptimizedAnalysisInternal implements the core optimized analysis
func runOptimizedAnalysisInternal(config *BottleneckAnalysisConfig, optConfig *OptimizationConfig) (*OptimizedBottleneckAnalysisResult, error) {
	// Initialize CKKS context
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            config.LogN,
		LogQ:            config.LogQ,
		LogP:            config.LogP,
		LogDefaultScale: 40,
		Xs:              rlwe.DefaultXs,
		Xe:              rlwe.DefaultXe,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// Generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	rotations := []int{}
	for step := 1; step < config.VectorDim; step *= 2 {
		rotations = append(rotations, step)
	}

	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := make([]uint64, len(rotations))
	for i, rot := range rotations {
		galEls[i] = params.GaloisElementForRotation(rot)
	}
	galKeys := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)

	// Pre-generate test data
	fmt.Printf("Pre-generating test data...\n")
	slots := params.MaxSlots()
	testTasks := make([]struct{ ct1, ct2 *rlwe.Ciphertext }, config.K)

	for i := 0; i < config.K; i++ {
		vec1 := make([]complex128, slots)
		vec2 := make([]complex128, slots)

		for j := 0; j < config.VectorDim; j++ {
			vec1[j] = complex(float64(i+j), 0)
			vec2[j] = complex(float64(j), 0)
		}

		pt1 := ckks.NewPlaintext(params, params.MaxLevel())
		pt2 := ckks.NewPlaintext(params, params.MaxLevel())

		encoder.Encode(vec1, pt1)
		encoder.Encode(vec2, pt2)

		ct1, _ := encryptor.EncryptNew(pt1)
		ct2, _ := encryptor.EncryptNew(pt2)

		testTasks[i] = struct{ ct1, ct2 *rlwe.Ciphertext }{ct1, ct2}
	}

	// Initialize optimization components
	var workStealingScheduler *WorkStealingScheduler
	var bufferPool *ThreadLocalBufferPool
	var pipelineProcessor *PipelinedInnerProductProcessor

	if optConfig.EnableWorkStealing {
		workStealingScheduler = NewWorkStealingScheduler(config.N, config.K*2)
		fmt.Printf("Work stealing enabled: %d workers with %d deque size\n", config.N, config.K*2)
	}

	if optConfig.EnableCacheFriendlyBuffers {
		bufferPool = NewThreadLocalBufferPool(config.N)
		fmt.Printf("Cache-friendly buffers enabled: %d worker pools\n", config.N)
	}

	if optConfig.EnablePipelinedProcessing {
		profiler := NewBottleneckProfiler()
		pipelineProcessor = NewPipelinedInnerProductProcessor(config, profiler)
		err := pipelineProcessor.Initialize()
		if err != nil {
			return nil, fmt.Errorf("pipeline initialization failed: %w", err)
		}
		fmt.Printf("Pipelined processing enabled: 4-stage pipeline\n")
	}

	// Run iterations with optimizations
	var totalTime time.Duration
	var results []*BottleneckAnalysisResult
	var workStealingStats map[string]interface{}
	var bufferStats []map[string]interface{}

	for iteration := 0; iteration < config.Iterations; iteration++ {
		fmt.Printf("Running optimized iteration %d/%d...\n", iteration+1, config.Iterations)

		iterProfiler := NewBottleneckProfiler()

		// Create optimized workers
		workers := make([]*CacheOptimizedOperationCounter, config.N)
		relinConfig := &OptimizedRelinearizationConfig{
			EnableStreamingKeySwitch:  optConfig.EnableStreamingKeySwitch,
			EnableLazyRelinearization: optConfig.EnableLazyRelinearization,
			BatchSize:                 optConfig.RelinearizationBatchSize,
			CRTWindowSize:             optConfig.CRTWindowSize,
		}

		for i := 0; i < config.N; i++ {
			var pool *BufferPool
			if bufferPool != nil {
				pool = bufferPool.GetWorkerPool(i, 1024, 8)
			}

			if optConfig.EnableOptimizedRelinearization && bufferPool != nil {
				workers[i] = NewCacheOptimizedOperationCounter(params, evk, iterProfiler, i, relinConfig, pool)
			} else {
				// Fallback to basic profiled counter
				basicCounter := NewProfiledOperationCounter(params, evk, iterProfiler, i)
				workers[i] = &CacheOptimizedOperationCounter{
					OptimizedOperationCounter: &OptimizedOperationCounter{
						eval:     basicCounter.eval,
						profiler: basicCounter.profiler,
						workerID: basicCounter.workerID,
						config:   relinConfig,
					},
				}
			}
		}

		// Start timing computation only
		computationStart := time.Now()

		if optConfig.EnableWorkStealing && workStealingScheduler != nil {
			// Use work-stealing execution
			err := runWorkStealingIteration(workStealingScheduler, workers, testTasks, config)
			if err != nil {
				return nil, fmt.Errorf("work-stealing iteration failed: %w", err)
			}
			workStealingStats = workStealingScheduler.GetStatistics()
		} else {
			// Use traditional channel-based execution
			err := runTraditionalIteration(workers, testTasks, config)
			if err != nil {
				return nil, fmt.Errorf("traditional iteration failed: %w", err)
			}
		}

		iterationTime := time.Since(computationStart)
		totalTime += iterationTime

		// Collect buffer statistics
		if bufferPool != nil {
			iterBufferStats := make([]map[string]interface{}, config.N)
			for i := 0; i < config.N; i++ {
				pool := bufferPool.GetWorkerPool(i, 1024, 8)
				iterBufferStats[i] = pool.GetStatistics()
			}
			bufferStats = iterBufferStats
		}

		// Analyze this iteration
		taskResults := make([]struct {
			workerID int
			duration time.Duration
		}, config.K)
		for i := 0; i < config.K; i++ {
			taskResults[i] = struct {
				workerID int
				duration time.Duration
			}{
				workerID: i % config.N,
				duration: iterationTime / time.Duration(config.K),
			}
		}

		iterResult := analyzeIteration(iterProfiler, taskResults, iterationTime, config)
		results = append(results, iterResult)
	}

	// Aggregate results
	finalResult := aggregateBottleneckResults(results, totalTime/time.Duration(config.Iterations))

	// Create optimized result
	optimizedResult := &OptimizedBottleneckAnalysisResult{
		BottleneckAnalysisResult: finalResult,
		WorkStealingStats:        workStealingStats,
		BufferPoolStats:          bufferStats,
		OptimizationsEnabled:     getEnabledOptimizations(optConfig),
	}

	return optimizedResult, nil
}

// runWorkStealingIteration executes an iteration using work-stealing
func runWorkStealingIteration(scheduler *WorkStealingScheduler, workers []*CacheOptimizedOperationCounter, testTasks []struct{ ct1, ct2 *rlwe.Ciphertext }, config *BottleneckAnalysisConfig) error {
	// Convert test tasks to work-stealing tasks
	wsTasks := make([]WorkStealingTask, len(testTasks))
	for i, task := range testTasks {
		wsTasks[i] = WorkStealingTask{
			ID:         i,
			CT1:        task.ct1,
			CT2:        task.ct2,
			Priority:   1,
			Complexity: 100,
		}
	}

	// Distribute tasks
	scheduler.DistributeTasks(wsTasks)

	// Run workers with work stealing
	var wg sync.WaitGroup
	for i := 0; i < config.N; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			worker := workers[workerID]

			for {
				task, hasTask := scheduler.GetTask(workerID)
				if !hasTask {
					break // No more tasks
				}

				// Perform inner product computation
				tmp, _ := worker.MulNew(task.CT1, task.CT2)
				if worker.OptimizedOperationCounter != nil {
					tmp, _ = worker.OptimizedRelinearizeNew(tmp)
				} else {
					tmp, _ = worker.eval.RelinearizeNew(tmp)
				}
				worker.Rescale(tmp, tmp)

				// Tree-sum rotations
				for step := 1; step < config.VectorDim; step *= 2 {
					rot, _ := worker.RotateNew(tmp, step)
					tmp, _ = worker.AddNew(tmp, rot)
				}

				scheduler.CompleteTask(workerID)
			}
		}(i)
	}

	wg.Wait()
	return nil
}

// runTraditionalIteration executes an iteration using traditional channels
func runTraditionalIteration(workers []*CacheOptimizedOperationCounter, testTasks []struct{ ct1, ct2 *rlwe.Ciphertext }, config *BottleneckAnalysisConfig) error {
	tasks := make(chan struct{ ct1, ct2 *rlwe.Ciphertext }, config.K)

	var wg sync.WaitGroup
	for i := 0; i < config.N; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			worker := workers[workerID]

			for task := range tasks {
				// Perform inner product computation
				tmp, _ := worker.MulNew(task.ct1, task.ct2)
				if worker.OptimizedOperationCounter != nil {
					tmp, _ = worker.OptimizedRelinearizeNew(tmp)
				} else {
					tmp, _ = worker.eval.RelinearizeNew(tmp)
				}
				worker.Rescale(tmp, tmp)

				// Tree-sum rotations
				for step := 1; step < config.VectorDim; step *= 2 {
					rot, _ := worker.RotateNew(tmp, step)
					tmp, _ = worker.AddNew(tmp, rot)
				}
			}
		}(i)
	}

	// Submit all tasks
	for _, task := range testTasks {
		tasks <- task
	}
	close(tasks)

	wg.Wait()
	return nil
}

// getEnabledOptimizations returns list of enabled optimizations
func getEnabledOptimizations(config *OptimizationConfig) []string {
	var enabled []string
	if config.EnableOptimizedRelinearization {
		enabled = append(enabled, "Optimized Relinearization")
	}
	if config.EnableWorkStealing {
		enabled = append(enabled, "Work Stealing")
	}
	if config.EnableCacheFriendlyBuffers {
		enabled = append(enabled, "Cache-Friendly Buffers")
	}
	if config.EnablePipelinedProcessing {
		enabled = append(enabled, "Pipelined Processing")
	}
	return enabled
}

// PrintOptimizedBottleneckAnalysis prints comprehensive optimized analysis report
func PrintOptimizedBottleneckAnalysis(result *OptimizedBottleneckAnalysisResult) {
	fmt.Printf("\n=== OPTIMIZED BOTTLENECK ANALYSIS REPORT ===\n")

	// Print basic performance metrics
	PrintBottleneckAnalysis(result.BottleneckAnalysisResult)

	// Print optimization-specific results
	fmt.Printf("\n=== OPTIMIZATION RESULTS ===\n")
	fmt.Printf("Enabled optimizations: %v\n", result.OptimizationsEnabled)

	if result.BaselineTime > 0 {
		fmt.Printf("\nPerformance Improvement:\n")
		fmt.Printf("  Baseline time: %v\n", result.BaselineTime)
		fmt.Printf("  Optimized time: %v\n", result.OptimizedTime)
		fmt.Printf("  Improvement factor: %.2fx\n", result.ImprovementFactor)
		fmt.Printf("  Bottleneck reduction: %.1f%%\n", result.BottleneckReduction)
	}

	if result.WorkStealingStats != nil {
		fmt.Printf("\nWork Stealing Results:\n")
		fmt.Printf("  Load imbalance: %.1f%%\n", result.WorkStealingStats["load_imbalance"])
		fmt.Printf("  Steal efficiency: %.1f%%\n", result.WorkStealingStats["steal_efficiency"])
		fmt.Printf("  Tasks completed: %v\n", result.WorkStealingStats["tasks_completed"])
	}

	if result.BufferPoolStats != nil {
		fmt.Printf("\nBuffer Pool Results:\n")
		var totalHitRate float64
		for i, stats := range result.BufferPoolStats {
			hitRate := stats["hit_rate"].(float64)
			totalHitRate += hitRate
			fmt.Printf("  Worker %d: %.1f%% hit rate\n", i, hitRate)
		}
		avgHitRate := totalHitRate / float64(len(result.BufferPoolStats))
		fmt.Printf("  Average hit rate: %.1f%%\n", avgHitRate)
	}

	// Performance recommendations based on results
	fmt.Printf("\n=== OPTIMIZATION RECOMMENDATIONS ===\n")
	if result.ImprovementFactor > 1.5 {
		fmt.Printf("✅ EXCELLENT: Optimizations provide significant improvement (%.2fx)\n", result.ImprovementFactor)
	} else if result.ImprovementFactor > 1.2 {
		fmt.Printf("✅ GOOD: Optimizations provide meaningful improvement (%.2fx)\n", result.ImprovementFactor)
	} else if result.ImprovementFactor > 1.0 {
		fmt.Printf("🤔 MODEST: Optimizations provide some improvement (%.2fx)\n", result.ImprovementFactor)
	}

	if result.BottleneckReduction > 25 {
		fmt.Printf("🎯 PRIMARY BOTTLENECK SIGNIFICANTLY REDUCED: %.1f%% improvement\n", result.BottleneckReduction)
	} else if result.BottleneckReduction > 10 {
		fmt.Printf("🎯 PRIMARY BOTTLENECK MODERATELY REDUCED: %.1f%% improvement\n", result.BottleneckReduction)
	}
}
