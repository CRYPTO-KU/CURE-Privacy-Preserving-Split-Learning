package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"cure_lib/core/ckkswrapper"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	logNs := []int{11, 12, 13, 14, 15, 16}
	numSamples := 64
	numCores := runtime.GOMAXPROCS(0)

	fmt.Printf("Benchmarking Parallel Encryption for %d samples using %d cores\n", numSamples, numCores)
	fmt.Printf("%-6s | %-10s | %-10s | %-20s | %-20s\n", "LogN", "Slots", "Encs", "Single Time", "Parallel Time")
	fmt.Println("--------------------------------------------------------------------------------")

	for _, logN := range logNs {
		// Define params based on LogN
		var paramsLit ckks.ParametersLiteral
		switch logN {
		case 11:
			paramsLit = ckks.ParametersLiteral{LogN: logN, LogQ: []int{28}, LogP: []int{25}, LogDefaultScale: 28}
		case 12:
			paramsLit = ckks.ParametersLiteral{LogN: logN, LogQ: []int{35, 35}, LogP: []int{35}, LogDefaultScale: 35}
		case 13:
			paramsLit = ckks.ParametersLiteral{LogN: logN, LogQ: []int{35, 35, 35, 35}, LogP: []int{35}, LogDefaultScale: 35}
		case 14:
			paramsLit = ckks.ParametersLiteral{LogN: logN, LogQ: []int{40, 40, 40, 40, 40}, LogP: []int{40}, LogDefaultScale: 40}
		case 15:
			paramsLit = ckks.ParametersLiteral{LogN: logN, LogQ: []int{60, 40, 40, 40, 38}, LogP: []int{60}, LogDefaultScale: 40}
		case 16:
			paramsLit = ckks.ParametersLiteral{LogN: logN, LogQ: []int{60, 50, 50, 50, 50, 50, 50}, LogP: []int{60}, LogDefaultScale: 50}
		}

		paramsLit.Xs = rlwe.DefaultXs
		paramsLit.Xe = rlwe.DefaultXe

		params, err := ckks.NewParametersFromLiteral(paramsLit)
		if err != nil {
			fmt.Printf("LogN=%d failed: %v\n", logN, err)
			continue
		}

		heCtx := ckkswrapper.NewHeContextWithParams(params)

		// Dummy data
		data := make([]float64, 784)
		for i := range data {
			data[i] = 0.5
		}

		pt := ckks.NewPlaintext(params, params.MaxLevel())
		heCtx.Encoder.Encode(data, pt)

		// Calculate Slots for display
		slots := 1 << (logN - 1)

		// Force 64 encryptions (no packing)
		encsNeeded := 64

		// Warmup
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		_ = heCtx.Decryptor.DecryptNew(ct)

		// Single-Core Encryption Benchmark
		startSingle := time.Now()
		for i := 0; i < encsNeeded; i++ {
			_, _ = heCtx.Encryptor.EncryptNew(pt)
		}
		singleTime := time.Since(startSingle)

		// Parallel Encryption Benchmark
		var wg sync.WaitGroup
		startTotal := time.Now()

		wg.Add(encsNeeded)
		for i := 0; i < encsNeeded; i++ {
			go func() {
				defer wg.Done()
				enc := heCtx.Encryptor.ShallowCopy()
				_, _ = enc.EncryptNew(pt)
			}()
		}
		wg.Wait()

		totalTime := time.Since(startTotal)

		fmt.Printf("%-6d | %-10d | %-10d | %-20v | %-20v\n", logN, slots, encsNeeded, singleTime, totalTime)
	}
}
