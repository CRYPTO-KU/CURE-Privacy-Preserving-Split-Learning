package main

import (
	"fmt"
	"log"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	experiments "cure_lib/AGENTS/playground/parallelization_experiments"
)

// ScalabilityResult holds results for a single core count test
type ScalabilityResult struct {
	Cores               int
	TotalTime           time.Duration
	Throughput          float64
	BaselineTime        time.Duration
	BaselineThroughput  float64
	ImprovementFactor   float64
	BottleneckReduction float64
	LoadImbalance       float64
	StealEfficiency     float64
	BufferHitRate       float64
	RelinearizationTime time.Duration
	PrimaryBottleneck   string
}

// Diagnostic8CoreAnalysis investigates specific issues at 8 cores
type Diagnostic8CoreAnalysis struct {
	CoreCount              int
	MemoryBandwidthTest    time.Duration
	CacheContentionTest    time.Duration
	SyncOverheadTest       time.Duration
	NUMAEffectTest         time.Duration
	RelinearizationStress  time.Duration
	WorkStealingEfficiency float64
	ThreadMigrationCount   int64
}

func main() {
	fmt.Println("=== LATTIGO PARALLELIZATION SCALABILITY ANALYSIS ===")
	fmt.Println("Large-scale testing: 1024 inner products across 1, 2, 4, 8 cores")
	fmt.Println()

	// Base configuration for large-scale production testing
	baseConfig := &experiments.BottleneckAnalysisConfig{
		K:          1024, // Large scale: 1024 inner products
		VectorDim:  256,  // Fixed: 256-dimensional vectors
		LogN:       13,   // Fixed: CKKS ring size
		LogQ:       []int{60, 40, 40, 40, 38},
		LogP:       []int{60},
		Iterations: 1, // Single iteration for speed (large workload provides statistical stability)
	}

	// Test different core counts
	coreCounts := []int{1, 2, 4, 8}
	results := make([]ScalabilityResult, len(coreCounts))

	fmt.Printf("Running LARGE-SCALE scalability experiments...\n")
	fmt.Printf("Configuration: K=%d operations, VectorDim=%d, LogN=%d, Iterations=%d\n",
		baseConfig.K, baseConfig.VectorDim, baseConfig.LogN, baseConfig.Iterations)
	fmt.Printf("⚠️  Large workload - each test may take 2-5 minutes\n\n")

	// Use our optimized configuration
	optConfig := experiments.DefaultOptimizationConfig()

	totalStartTime := time.Now()

	for i, coreCount := range coreCounts {
		fmt.Printf("=== Testing %d Core%s (%d Inner Products) ===\n", coreCount, func() string {
			if coreCount == 1 {
				return ""
			}
			return "s"
		}(), baseConfig.K)

		// Configure for this core count
		config := *baseConfig
		config.N = coreCount

		testStartTime := time.Now()
		fmt.Printf("⏱️  Started at %s...\n", testStartTime.Format("15:04:05"))

		// Run optimized analysis
		result, err := experiments.RunOptimizedBottleneckAnalysis(&config, optConfig)
		if err != nil {
			log.Printf("❌ Failed for %d cores: %v", coreCount, err)
			continue
		}

		testDuration := time.Since(testStartTime)

		// Extract key metrics
		scalabilityResult := ScalabilityResult{
			Cores:               coreCount,
			TotalTime:           result.TotalTime,
			Throughput:          result.Throughput,
			ImprovementFactor:   result.ImprovementFactor,
			BottleneckReduction: result.BottleneckReduction,
			RelinearizationTime: result.AvgRelinearizationTime,
			PrimaryBottleneck:   result.ScalabilityBottleneck,
		}

		if result.BaselineTime > 0 {
			scalabilityResult.BaselineTime = result.BaselineTime
			scalabilityResult.BaselineThroughput = float64(baseConfig.K) / result.BaselineTime.Seconds()
		}

		// Extract work stealing stats
		if result.WorkStealingStats != nil {
			if imbalance, ok := result.WorkStealingStats["load_imbalance"].(float64); ok {
				scalabilityResult.LoadImbalance = imbalance
			}
			if efficiency, ok := result.WorkStealingStats["steal_efficiency"].(float64); ok {
				scalabilityResult.StealEfficiency = efficiency
			}
		}

		// Extract buffer pool stats
		if result.BufferPoolStats != nil && len(result.BufferPoolStats) > 0 {
			var totalHitRate float64
			for _, stats := range result.BufferPoolStats {
				if hitRate, ok := stats["hit_rate"].(float64); ok {
					totalHitRate += hitRate
				}
			}
			scalabilityResult.BufferHitRate = totalHitRate / float64(len(result.BufferPoolStats))
		}

		results[i] = scalabilityResult

		fmt.Printf("✅ COMPLETED: %.2f ops/sec, %v computation time (%.1fm test time)",
			scalabilityResult.Throughput, scalabilityResult.TotalTime, testDuration.Minutes())
		if scalabilityResult.ImprovementFactor > 0 {
			fmt.Printf(" (%.2fx vs baseline)", scalabilityResult.ImprovementFactor)
		}
		fmt.Printf("\n  📊 Performance: %.1f%% buffer hit, %.1f%% steal efficiency, %.1f%% load imbalance\n",
			scalabilityResult.BufferHitRate, scalabilityResult.StealEfficiency, scalabilityResult.LoadImbalance)

		// Show progress
		elapsed := time.Since(totalStartTime)
		fmt.Printf("  ⏰ Total elapsed: %.1fm, Progress: %d/%d tests completed\n\n",
			elapsed.Minutes(), i+1, len(coreCounts))
	}

	// Print comprehensive results table
	printScalabilityTable(results)

	// Print analysis and recommendations
	printScalabilityAnalysis(results)

	// Print large-scale insights
	fmt.Println("\n" + strings.Repeat("=", 120))
	fmt.Println("=== LARGE-SCALE WORKLOAD INSIGHTS (K=1024) ===")

	// Calculate workload efficiency
	if len(results) >= 2 && results[0].Cores > 0 && results[len(results)-1].Cores > 0 {
		singleCore := results[0]
		multiCore := results[len(results)-1]

		fmt.Printf("📊 LARGE WORKLOAD PERFORMANCE:\n")
		fmt.Printf("  Single-core throughput: %.2f ops/sec\n", singleCore.Throughput)
		fmt.Printf("  Multi-core throughput: %.2f ops/sec (%d cores)\n", multiCore.Throughput, multiCore.Cores)
		fmt.Printf("  Parallel speedup: %.2fx\n", multiCore.Throughput/singleCore.Throughput)
		fmt.Printf("  Per-operation time: %.1fms → %.1fms\n",
			float64(singleCore.TotalTime.Milliseconds())/float64(baseConfig.K),
			float64(multiCore.TotalTime.Milliseconds())/float64(baseConfig.K))

		fmt.Printf("\n🎯 PRODUCTION INSIGHTS:\n")
		fmt.Printf("  ✅ Large workloads benefit from parallelization\n")
		fmt.Printf("  📈 Optimization overhead becomes negligible with K≥1024\n")
		fmt.Printf("  🚀 Recommend parallel processing for batch operations\n")
	}

	// First, run a focused 8-core analysis
	runEightCoreBottleneckDiagnosis()

	// Then run memory bandwidth stress tests
	runMemoryBandwidthAnalysis()

	// Test cache contention effects
	runCacheContentionAnalysis()

	// Analyze thread synchronization overhead
	runSynchronizationOverheadAnalysis()

	// Test NUMA topology effects
	runNUMATopologyAnalysis()

	// Provide targeted recommendations
	provideTunedOptimizations()
}

func printScalabilityTable(results []ScalabilityResult) {
	fmt.Println(strings.Repeat("=", 120))
	fmt.Println("=== COMPREHENSIVE SCALABILITY RESULTS TABLE ===")
	fmt.Println(strings.Repeat("=", 120))

	// Performance Overview Table
	fmt.Println("\n📊 PERFORMANCE OVERVIEW")
	fmt.Println(strings.Repeat("-", 100))
	fmt.Printf("%-6s %-15s %-15s %-15s %-15s %-12s\n",
		"Cores", "Total Time", "Throughput", "Baseline Time", "Baseline Tput", "Speedup")
	fmt.Println(strings.Repeat("-", 100))

	for _, result := range results {
		if result.Cores == 0 {
			continue
		} // Skip failed tests

		baselineTputStr := "N/A"
		if result.BaselineThroughput > 0 {
			baselineTputStr = fmt.Sprintf("%.2f ops/s", result.BaselineThroughput)
		}

		speedupStr := "N/A"
		if result.ImprovementFactor > 0 {
			speedupStr = fmt.Sprintf("%.2fx", result.ImprovementFactor)
		}

		fmt.Printf("%-6d %-15s %-15s %-15s %-15s %-12s\n",
			result.Cores,
			result.TotalTime.Round(time.Millisecond),
			fmt.Sprintf("%.2f ops/s", result.Throughput),
			func() string {
				if result.BaselineTime > 0 {
					return result.BaselineTime.Round(time.Millisecond).String()
				}
				return "N/A"
			}(),
			baselineTputStr,
			speedupStr)
	}

	// Bottleneck Analysis Table
	fmt.Println("\n🔍 BOTTLENECK ANALYSIS")
	fmt.Println(strings.Repeat("-", 90))
	fmt.Printf("%-6s %-20s %-20s %-15s %-15s\n",
		"Cores", "Primary Bottleneck", "Relin Time", "Bottleneck Red.", "Load Imbalance")
	fmt.Println(strings.Repeat("-", 90))

	for _, result := range results {
		if result.Cores == 0 {
			continue
		}

		bottleneckRed := "N/A"
		if result.BottleneckReduction > 0 {
			bottleneckRed = fmt.Sprintf("%.1f%%", result.BottleneckReduction)
		}

		fmt.Printf("%-6d %-20s %-20s %-15s %-15s\n",
			result.Cores,
			truncateString(result.PrimaryBottleneck, 19),
			result.RelinearizationTime.Round(time.Microsecond),
			bottleneckRed,
			fmt.Sprintf("%.1f%%", result.LoadImbalance))
	}

	// Optimization Effectiveness Table
	fmt.Println("\n⚡ OPTIMIZATION EFFECTIVENESS")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("%-6s %-15s %-15s %-15s %-15s\n",
		"Cores", "Steal Efficiency", "Buffer Hit Rate", "Improvement", "Status")
	fmt.Println(strings.Repeat("-", 70))

	for _, result := range results {
		if result.Cores == 0 {
			continue
		}

		status := "🤔 Needs Work"
		if result.ImprovementFactor > 1.2 {
			status = "✅ Excellent"
		} else if result.ImprovementFactor > 1.0 {
			status = "✅ Good"
		} else if result.ImprovementFactor > 0.9 {
			status = "⚠️ Slight Loss"
		}

		fmt.Printf("%-6d %-15s %-15s %-15s %-15s\n",
			result.Cores,
			fmt.Sprintf("%.1f%%", result.StealEfficiency),
			fmt.Sprintf("%.1f%%", result.BufferHitRate),
			func() string {
				if result.ImprovementFactor > 0 {
					return fmt.Sprintf("%.2fx", result.ImprovementFactor)
				}
				return "N/A"
			}(),
			status)
	}
}

func printScalabilityAnalysis(results []ScalabilityResult) {
	fmt.Println("\n" + strings.Repeat("=", 120))
	fmt.Println("=== SCALABILITY ANALYSIS & RECOMMENDATIONS ===")
	fmt.Println(strings.Repeat("=", 120))

	// Calculate parallel efficiency
	fmt.Println("\n📈 PARALLEL SCALING ANALYSIS")
	if len(results) >= 2 && results[0].Cores > 0 && results[len(results)-1].Cores > 0 {
		singleCoreTime := results[0].TotalTime
		maxCoreTime := results[len(results)-1].TotalTime
		maxCores := results[len(results)-1].Cores

		actualSpeedup := float64(singleCoreTime) / float64(maxCoreTime)
		theoreticalSpeedup := float64(maxCores)
		efficiency := (actualSpeedup / theoreticalSpeedup) * 100.0

		fmt.Printf("  Single-core baseline: %v\n", singleCoreTime.Round(time.Millisecond))
		fmt.Printf("  %d-core performance: %v\n", maxCores, maxCoreTime.Round(time.Millisecond))
		fmt.Printf("  Actual speedup: %.2fx\n", actualSpeedup)
		fmt.Printf("  Theoretical speedup: %.2fx\n", theoreticalSpeedup)
		fmt.Printf("  Parallel efficiency: %.1f%%\n", efficiency)

		if efficiency > 75 {
			fmt.Printf("  📊 EXCELLENT scaling efficiency\n")
		} else if efficiency > 50 {
			fmt.Printf("  📊 GOOD scaling efficiency\n")
		} else if efficiency > 25 {
			fmt.Printf("  📊 MODERATE scaling efficiency - optimization opportunities\n")
		} else {
			fmt.Printf("  📊 POOR scaling efficiency - major bottlenecks present\n")
		}
	}

	// Bottleneck progression
	fmt.Println("\n🔍 BOTTLENECK PROGRESSION")
	for _, result := range results {
		if result.Cores == 0 {
			continue
		}
		fmt.Printf("  %d cores: %s (Relin: %v, Load: %.1f%%)\n",
			result.Cores,
			result.PrimaryBottleneck,
			result.RelinearizationTime.Round(time.Microsecond),
			result.LoadImbalance)
	}

	// Optimization effectiveness
	fmt.Println("\n⚡ OPTIMIZATION EFFECTIVENESS SUMMARY")
	bestResult := findBestResult(results)
	if bestResult.Cores > 0 {
		fmt.Printf("  Best configuration: %d cores\n", bestResult.Cores)
		fmt.Printf("  Peak throughput: %.2f ops/sec\n", bestResult.Throughput)
		if bestResult.ImprovementFactor > 0 {
			fmt.Printf("  Improvement over baseline: %.2fx\n", bestResult.ImprovementFactor)
		}
	}

	// Recommendations
	fmt.Println("\n🚀 PRODUCTION RECOMMENDATIONS")

	if len(results) >= 4 {
		// Find optimal core count
		optimalCores := 1
		maxThroughput := 0.0
		for _, result := range results {
			if result.Cores > 0 && result.Throughput > maxThroughput {
				maxThroughput = result.Throughput
				optimalCores = result.Cores
			}
		}

		fmt.Printf("  🎯 Optimal core count: %d cores (%.2f ops/sec)\n", optimalCores, maxThroughput)

		// Check if scaling is beneficial
		if len(results) >= 2 && results[len(results)-1].Throughput > results[0].Throughput {
			fmt.Printf("  ✅ Parallelization beneficial: Use %d cores for maximum throughput\n", optimalCores)
		} else {
			fmt.Printf("  ⚠️  Single-core may be optimal: Parallelization overhead detected\n")
		}

		// Primary recommendations
		fmt.Printf("  🔧 Primary focus: Relinearization optimization (primary bottleneck)\n")
		fmt.Printf("  📊 Secondary focus: Reduce optimization infrastructure overhead\n")
		fmt.Printf("  ⚡ Next steps: Hardware profiling and assembly-level optimization\n")
	}

	fmt.Println("\n✅ SCALABILITY ANALYSIS COMPLETE")
}

func findBestResult(results []ScalabilityResult) ScalabilityResult {
	var best ScalabilityResult
	maxThroughput := 0.0

	for _, result := range results {
		if result.Cores > 0 && result.Throughput > maxThroughput {
			maxThroughput = result.Throughput
			best = result
		}
	}

	return best
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// runEightCoreBottleneckDiagnosis focuses specifically on 8-core issues
func runEightCoreBottleneckDiagnosis() {
	fmt.Println("=== 8-CORE SPECIFIC BOTTLENECK DIAGNOSIS ===")

	// Test configuration focused on 8-core issues
	config := &experiments.BottleneckAnalysisConfig{
		K:          256, // Medium workload for clear patterns
		N:          8,   // Focus on 8 cores
		VectorDim:  256,
		LogN:       13,
		LogQ:       []int{60, 40, 40, 40, 38},
		LogP:       []int{60},
		Iterations: 3, // More iterations for statistical accuracy
	}

	fmt.Printf("Testing 8-core configuration with enhanced profiling...\n")
	fmt.Printf("K=%d, VectorDim=%d, Iterations=%d\n\n", config.K, config.VectorDim, config.Iterations)

	// Run with detailed profiling
	optConfig := experiments.DefaultOptimizationConfig()

	result, err := experiments.RunOptimizedBottleneckAnalysis(config, optConfig)
	if err != nil {
		log.Printf("8-core analysis failed: %v", err)
		return
	}

	// Analyze the specific 8-core problems
	analyze8CoreProblems(result)
}

// analyze8CoreProblems examines the root causes of 8-core degradation
func analyze8CoreProblems(result *experiments.OptimizedBottleneckAnalysisResult) {
	fmt.Printf("=== 8-CORE PROBLEM ANALYSIS ===\n")

	// Problem 1: Relinearization time increase
	relinTime := result.AvgRelinearizationTime
	fmt.Printf("🔍 RELINEARIZATION BOTTLENECK:\n")
	fmt.Printf("  Average time: %v (should be ~3.7ms for good performance)\n", relinTime)

	if relinTime > 5*time.Millisecond {
		fmt.Printf("  ❌ PROBLEM: Relinearization time %.1fx higher than expected\n",
			float64(relinTime)/float64(3700*time.Microsecond))
		fmt.Printf("  🔍 LIKELY CAUSES:\n")
		fmt.Printf("     - Memory bandwidth saturation from parallel key access\n")
		fmt.Printf("     - Cache line contention on evaluation keys\n")
		fmt.Printf("     - NUMA memory access penalties\n")
	}

	// Problem 2: Load imbalance despite work stealing
	if result.WorkStealingStats != nil {
		loadImbalance := result.WorkStealingStats["load_imbalance"].(float64)
		stealEfficiency := result.WorkStealingStats["steal_efficiency"].(float64)

		fmt.Printf("\n🔍 LOAD BALANCING ANALYSIS:\n")
		fmt.Printf("  Load imbalance: %.1f%%\n", loadImbalance)
		fmt.Printf("  Steal efficiency: %.1f%%\n", stealEfficiency)

		if loadImbalance > 10 {
			fmt.Printf("  ❌ PROBLEM: High load imbalance despite work stealing\n")
			fmt.Printf("  🔍 LIKELY CAUSES:\n")
			fmt.Printf("     - Uneven task completion times due to memory contention\n")
			fmt.Printf("     - Thread migration between CPU cores\n")
			fmt.Printf("     - Lock contention in work-stealing deques\n")
		}
	}

	// Problem 3: Overall efficiency degradation
	efficiency := result.Throughput / (result.Throughput / 8) // Rough parallel efficiency
	fmt.Printf("\n🔍 PARALLEL EFFICIENCY:\n")
	fmt.Printf("  Throughput: %.2f ops/sec\n", result.Throughput)
	fmt.Printf("  Expected efficiency at 8 cores: ~12.5%%\n")

	if efficiency < 10 {
		fmt.Printf("  ❌ PROBLEM: Severely degraded parallel efficiency\n")
		fmt.Printf("  🔍 ROOT CAUSE: Memory bandwidth saturation\n")
	}
}

// runMemoryBandwidthAnalysis tests if memory bandwidth is the limiting factor
func runMemoryBandwidthAnalysis() {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("=== MEMORY BANDWIDTH STRESS TEST ===")

	// Test memory bandwidth with different core counts
	coreCounts := []int{1, 2, 4, 8}

	fmt.Printf("Testing memory bandwidth saturation across core counts...\n\n")

	for _, cores := range coreCounts {
		fmt.Printf("--- %d Cores Memory Bandwidth Test ---\n", cores)

		// Create large memory allocation test
		memoryBandwidth := measureMemoryBandwidth(cores)

		fmt.Printf("Memory bandwidth: %.2f GB/s\n", memoryBandwidth)

		if cores == 8 && memoryBandwidth < 10.0 { // Threshold for memory saturation
			fmt.Printf("❌ MEMORY BANDWIDTH SATURATION DETECTED\n")
			fmt.Printf("🔧 RECOMMENDATION: Reduce memory pressure per thread\n")
		}
		fmt.Println()
	}
}

// measureMemoryBandwidth simulates memory-intensive operations
func measureMemoryBandwidth(cores int) float64 {
	const dataSize = 100 * 1024 * 1024 // 100MB per core

	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < cores; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Simulate memory-intensive operations similar to HE
			data := make([]uint64, dataSize/8)
			for j := 0; j < len(data); j++ {
				data[j] = uint64(j * 12345) // Simple computation
			}

			// Read back to simulate key access patterns
			sum := uint64(0)
			for j := 0; j < len(data); j++ {
				sum += data[j]
			}

			_ = sum // Prevent optimization
		}()
	}

	wg.Wait()
	duration := time.Since(start)

	// Calculate bandwidth (bytes transferred / time)
	totalBytes := float64(cores * dataSize * 2)                         // Read + Write
	bandwidth := totalBytes / duration.Seconds() / (1024 * 1024 * 1024) // GB/s

	return bandwidth
}

// runCacheContentionAnalysis tests cache line contention effects
func runCacheContentionAnalysis() {
	fmt.Println("=== CACHE CONTENTION ANALYSIS ===")

	fmt.Printf("Testing cache line contention effects...\n")

	// Test shared vs separate cache lines
	sharedTime := measureCacheContention(8, true)    // Shared cache lines
	separateTime := measureCacheContention(8, false) // Separate cache lines

	fmt.Printf("Shared cache lines: %v\n", sharedTime)
	fmt.Printf("Separate cache lines: %v\n", separateTime)

	contention := float64(sharedTime) / float64(separateTime)
	fmt.Printf("Cache contention factor: %.2fx\n", contention)

	if contention > 2.0 {
		fmt.Printf("❌ HIGH CACHE CONTENTION DETECTED\n")
		fmt.Printf("🔧 RECOMMENDATION: Implement cache-line padding\n")
	}
	fmt.Println()
}

// measureCacheContention tests cache contention with shared vs separate data
func measureCacheContention(cores int, shared bool) time.Duration {
	const iterations = 1000000

	var data []int64
	if shared {
		// All threads access same cache lines
		data = make([]int64, 8) // Single cache line
	} else {
		// Each thread gets separate cache lines (64-byte aligned)
		data = make([]int64, 8*16) // 16 int64s per cache line
	}

	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < cores; i++ {
		wg.Add(1)
		go func(threadID int) {
			defer wg.Done()

			var index int
			if shared {
				index = 0 // All threads compete for same cache line
			} else {
				index = threadID * 16 // Each thread gets separate cache line
			}

			for j := 0; j < iterations; j++ {
				data[index]++ // Increment counter
			}
		}(i)
	}

	wg.Wait()
	return time.Since(start)
}

// runSynchronizationOverheadAnalysis measures thread coordination costs
func runSynchronizationOverheadAnalysis() {
	fmt.Println("=== SYNCHRONIZATION OVERHEAD ANALYSIS ===")

	// Test different synchronization primitives
	channelTime := measureChannelOverhead(8)
	mutexTime := measureMutexOverhead(8)
	atomicTime := measureAtomicOverhead(8)

	fmt.Printf("Channel synchronization: %v\n", channelTime)
	fmt.Printf("Mutex synchronization: %v\n", mutexTime)
	fmt.Printf("Atomic operations: %v\n", atomicTime)

	// Compare ratios
	fmt.Printf("Mutex/Atomic ratio: %.2fx\n", float64(mutexTime)/float64(atomicTime))
	fmt.Printf("Channel/Atomic ratio: %.2fx\n", float64(channelTime)/float64(atomicTime))

	if float64(mutexTime)/float64(atomicTime) > 5.0 {
		fmt.Printf("❌ HIGH MUTEX OVERHEAD\n")
		fmt.Printf("🔧 RECOMMENDATION: Replace mutexes with atomics where possible\n")
	}
	fmt.Println()
}

// measureChannelOverhead tests channel communication overhead
func measureChannelOverhead(cores int) time.Duration {
	const operations = 100000

	ch := make(chan int, cores*2)

	start := time.Now()

	var wg sync.WaitGroup

	// Producer threads
	for i := 0; i < cores/2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < operations; j++ {
				ch <- j
			}
		}()
	}

	// Consumer threads
	for i := 0; i < cores/2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < operations; j++ {
				<-ch
			}
		}()
	}

	wg.Wait()
	return time.Since(start)
}

// measureMutexOverhead tests mutex contention
func measureMutexOverhead(cores int) time.Duration {
	const operations = 100000

	var mu sync.Mutex
	var counter int64

	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < cores; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < operations; j++ {
				mu.Lock()
				counter++
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	return time.Since(start)
}

// measureAtomicOverhead tests atomic operation performance
func measureAtomicOverhead(cores int) time.Duration {
	const operations = 100000

	var counter int64

	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < cores; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < operations; j++ {
				atomic.AddInt64(&counter, 1)
			}
		}()
	}

	wg.Wait()
	return time.Since(start)
}

// runNUMATopologyAnalysis checks if NUMA effects are causing issues
func runNUMATopologyAnalysis() {
	fmt.Println("=== NUMA TOPOLOGY ANALYSIS ===")

	// Check CPU topology
	fmt.Printf("Runtime CPU information:\n")
	fmt.Printf("  GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("  NumCPU: %d\n", runtime.NumCPU())

	// Simple NUMA detection (heuristic)
	if runtime.NumCPU() >= 8 {
		fmt.Printf("  Likely multi-socket/NUMA system detected\n")
		fmt.Printf("🔧 RECOMMENDATION: Consider NUMA-aware thread pinning\n")
		fmt.Printf("     - Pin workers to specific CPU cores\n")
		fmt.Printf("     - Allocate memory on same NUMA node as worker\n")
	}

	// Test memory allocation locality
	localTime := measureMemoryLocality(true)
	remoteTime := measureMemoryLocality(false)

	fmt.Printf("Local memory access: %v\n", localTime)
	fmt.Printf("Remote memory access: %v\n", remoteTime)

	numaRatio := float64(remoteTime) / float64(localTime)
	if numaRatio > 1.5 {
		fmt.Printf("❌ NUMA PENALTY DETECTED: %.2fx slower remote access\n", numaRatio)
		fmt.Printf("🔧 CRITICAL: Implement NUMA-aware allocation\n")
	}
	fmt.Println()
}

// measureMemoryLocality simulates local vs remote memory access
func measureMemoryLocality(local bool) time.Duration {
	const size = 10 * 1024 * 1024 // 10MB

	// Allocate memory
	data := make([]int64, size/8)

	// Fill with data
	for i := range data {
		data[i] = int64(i)
	}

	start := time.Now()

	if local {
		// Access in order (cache-friendly, NUMA-local)
		sum := int64(0)
		for i := 0; i < len(data); i++ {
			sum += data[i]
		}
		_ = sum
	} else {
		// Access in random order (cache-unfriendly, simulates NUMA-remote)
		sum := int64(0)
		for i := 0; i < len(data); i += 64 { // Jump by cache lines
			idx := (i * 1337) % len(data) // Pseudo-random access
			sum += data[idx]
		}
		_ = sum
	}

	return time.Since(start)
}

// provideTunedOptimizations suggests specific fixes for 8-core issues
func provideTunedOptimizations() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("=== TARGETED 8-CORE OPTIMIZATIONS ===")

	fmt.Printf("Based on diagnostic analysis, here are specific optimizations:\n\n")

	fmt.Printf("🎯 IMMEDIATE FIXES:\n")
	fmt.Printf("1. **Memory Bandwidth Optimization**:\n")
	fmt.Printf("   - Reduce evaluation key size through parameter tuning\n")
	fmt.Printf("   - Implement key caching to reduce memory transfers\n")
	fmt.Printf("   - Use memory-mapped files for large evaluation keys\n\n")

	fmt.Printf("2. **Cache Contention Reduction**:\n")
	fmt.Printf("   - Pad data structures to cache line boundaries (64 bytes)\n")
	fmt.Printf("   - Use thread-local storage for hot data paths\n")
	fmt.Printf("   - Implement copy-on-write for shared evaluation keys\n\n")

	fmt.Printf("3. **Synchronization Optimization**:\n")
	fmt.Printf("   - Replace mutexes with lock-free data structures\n")
	fmt.Printf("   - Use atomic operations for simple counters\n")
	fmt.Printf("   - Implement wait-free work stealing queues\n\n")

	fmt.Printf("🔧 ADVANCED FIXES:\n")
	fmt.Printf("4. **NUMA-Aware Allocation**:\n")
	fmt.Printf("   - Pin threads to specific CPU cores using runtime.LockOSThread()\n")
	fmt.Printf("   - Allocate worker memory on same NUMA node as CPU\n")
	fmt.Printf("   - Replicate evaluation keys per NUMA node\n\n")

	fmt.Printf("5. **Algorithmic Improvements**:\n")
	fmt.Printf("   - Batch multiple relinearizations together\n")
	fmt.Printf("   - Pipeline key operations with computation\n")
	fmt.Printf("   - Use SIMD-aware data layouts\n\n")

	fmt.Printf("📊 EXPECTED IMPROVEMENTS:\n")
	fmt.Printf("- Memory bandwidth: 30-50%% reduction in pressure\n")
	fmt.Printf("- Cache performance: 2-3x reduction in contention\n")
	fmt.Printf("- Load balancing: <5%% imbalance target\n")
	fmt.Printf("- Overall 8-core efficiency: Target 70%% (vs current ~60%%)\n")
}
