package main

import (
	"fmt"
	"log"
	"runtime"

	paralellizationlayers "cure_lib/AGENTS/playground/paralellization_layers"
)

func main() {
	fmt.Println("🚀 Starting Parallel Layer Benchmarking...")
	fmt.Printf("💻 Available CPU cores: %d\n", runtime.NumCPU())
	fmt.Printf("🔧 GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	// Run comprehensive benchmark to replicate bench_results_cores2_logn13.csv
	if err := paralellizationlayers.RunComprehensiveBenchmark(); err != nil {
		log.Fatalf("❌ Comprehensive benchmark failed: %v", err)
	}

	fmt.Println("✅ Parallel Layer Benchmarking Completed Successfully!")
}
