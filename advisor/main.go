package main

import (
	"flag"
	"fmt"
	"runtime"
	"sync"
	"time"
)

const (
	LogN = 16
)

func main() {

	var num_CPU int
	flag.IntVar(&num_CPU, "cpu", runtime.NumCPU(), "number of CPUs to use")
	flag.Parse()

	runtime.GOMAXPROCS(num_CPU)
	fmt.Printf("Number of CPUs used: %d\n", num_CPU)

	cryptoSystem := InitCryptoSystem()

	netArch := NewNetworkArchitecture([]int{784, 128, 32, 10}, 2)
	desiredTrainingTime := 100000 * time.Second
	sizeOfCiphertext := 0.0078125
	networkRate := 1.0

	var wg sync.WaitGroup
	batch_size := 120

	cts := generateCiphertexts(batch_size, cryptoSystem)

	rotationTime := measureRotationTime(cts, &wg, num_CPU, *cryptoSystem.evaluator)
	degreeOfActivationPolynomial := 3
	maxDesiredDepth := 15
	maxMemory := 500.0
	maxClientTime := 2 * time.Second
	numSamples := 10000

	lastSplitIdx, clientTime, clientMemoryUsage := clientEstimator(netArch, maxMemory, maxClientTime, numSamples)

	fmt.Println("----- Client Estimator Part -----")
	fmt.Println("Last successful split index:", lastSplitIdx)
	fmt.Println("Client time taken:", clientTime)
	fmt.Printf("Client memory usage: %.2f MB\n", clientMemoryUsage)

	lastSplitIdx, serverTime, serverOperationsDepth := serverEstimator(rotationTime, netArch, desiredTrainingTime, numSamples, LogN, sizeOfCiphertext, networkRate, degreeOfActivationPolynomial, maxDesiredDepth)

	fmt.Println("----- Server Estimator Part -----")
	fmt.Println("Last successful split index:", lastSplitIdx)
	fmt.Println("Server time taken:", serverTime)
	fmt.Printf("Server operations depth: %.2f\n", serverOperationsDepth)

	validSplitIndices, estimatedTimes, _ := advisor(rotationTime, netArch, desiredTrainingTime, numSamples, LogN, sizeOfCiphertext, networkRate, degreeOfActivationPolynomial, maxDesiredDepth, maxMemory, maxClientTime, cryptoSystem.params, *cryptoSystem.encoder, *cryptoSystem.encryptor, cryptoSystem.evl)
	for i := 0; i < len(validSplitIndices); i++ {
		fmt.Printf("Split Index: %d, Estimated Time: %v\n", validSplitIndices[i], estimatedTimes[i])
	}
}
