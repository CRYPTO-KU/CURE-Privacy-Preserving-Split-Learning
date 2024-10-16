package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
)

func estimate(rotationTime time.Duration, netArch *NetworkArchitecture, splitIdx int, numSamples, LogN int, sizeOfCiphertext float64, networkRate float64) time.Duration {
	serverLayers := netArch.Layers[:splitIdx]
	//clientLayers := netArch.Layers[splitIdx:]

	Ns := 8192 // LogN Number of slots
	var totalRotations int

	for i := 0; i < len(serverLayers)-1; i++ {
		sizeLi := SmallestPowerOfTwo(serverLayers[i])
		sizeLiPlus1 := SmallestPowerOfTwo(serverLayers[i+1])
		rotationsForLayer := numSamples * sizeLi * sizeLiPlus1 / (Ns * 2) * int(math.Log2(float64(sizeLiPlus1)))
		totalRotations += rotationsForLayer
	}

	totalEstimatedRotationTime := time.Duration(totalRotations) * rotationTime

	lastLayerSize := serverLayers[len(serverLayers)-1]
	dataTransferSize := CalculateDataTransfer(numSamples, lastLayerSize, Ns, sizeOfCiphertext)
	dataTransferTime := CalculateNetworkTime(dataTransferSize, networkRate)

	totalEstimatedTime := totalEstimatedRotationTime + dataTransferTime

	return totalEstimatedTime
}

func clientEstimator(netArch *NetworkArchitecture, maxMemory float64, maxClientTime time.Duration, numSamples int) (lastSplitIdx int, clientTime time.Duration, clientMemoryUsage float64) {
	lastSplitIdx = len(netArch.Layers) // Start with the split index at the end
	clientTime = 0
	clientMemoryUsage = 0

	for i := len(netArch.Layers) - 1; i >= 0; i-- {
		// Set the split index
		netArch.SplitIdx = i

		// Calculate client memory usage and client-side computation time
		clientLayers := netArch.Layers[i:]
		clientMemoryUsage = estimateClientMemoryUsage(clientLayers, numSamples)

		// Perform the microbenchmark for the dot product
		averageDotProductTime := microBenchmarkDotProduct(clientLayers, numSamples)

		// Estimate the client time with the benchmarked dot product time
		clientTime = averageDotProductTime * time.Duration(numSamples)

		// Check if the thresholds are satisfied
		if clientMemoryUsage <= maxMemory && clientTime <= maxClientTime {
			lastSplitIdx = i
		} else {
			// If thresholds exceeded, break and return the last successful split index
			break
		}
	}

	return lastSplitIdx, clientTime, clientMemoryUsage
}

func serverEstimator(rotationTime time.Duration, netArch *NetworkArchitecture, desiredTime time.Duration, numSamples, LogN int, sizeOfCiphertext float64, networkRate float64, degreeOfActivationPolynomial int, desiredDepth int) (lastSplitIdx int, serverTime time.Duration, serverOperationsDepth float64) {
	var maxDepthIdx int

	// First pass: Find the maximum index where the depth of operations is within the desired depth
	for splitIdx := 1; splitIdx < len(netArch.Layers); splitIdx++ {
		serverLayers := netArch.Layers[:splitIdx]

		// Calculate the depth of operations for the server-side layers
		matrixMultiplicationDepth := float64(len(serverLayers) - 1)
		activationFunctionDepth := float64((len(serverLayers) - 2) * degreeOfActivationPolynomial)
		totalDepth := matrixMultiplicationDepth + activationFunctionDepth

		// Check if the depth is within the desired limit
		if int(totalDepth) <= desiredDepth {
			maxDepthIdx = splitIdx
		} else {
			break
		}
	}

	// Second pass: Check the time constraints up to the maxDepthIdx
	for splitIdx := 1; splitIdx <= maxDepthIdx; splitIdx++ {
		serverLayers := netArch.Layers[:splitIdx]

		// Calculate the depth of operations for the server-side layers
		matrixMultiplicationDepth := float64(len(serverLayers) - 1)
		activationFunctionDepth := float64((len(serverLayers) - 2) * degreeOfActivationPolynomial)
		totalDepth := matrixMultiplicationDepth + activationFunctionDepth

		// Estimate the total rotation time for the server-side layers
		totalRotations := 0
		Ns := 8192 // Number of slots
		for i := 0; i < len(serverLayers)-1; i++ {
			sizeLi := SmallestPowerOfTwo(serverLayers[i])
			sizeLiPlus1 := SmallestPowerOfTwo(serverLayers[i+1])
			rotationsForLayer := numSamples * sizeLi * sizeLiPlus1 / (Ns * 2) * int(math.Log2(float64(sizeLiPlus1)))
			totalRotations += rotationsForLayer
		}
		totalEstimatedRotationTime := time.Duration(totalRotations) * rotationTime

		// Estimate the data transfer time after the server-side layers
		lastLayerSize := serverLayers[len(serverLayers)-1]
		dataTransferSize := CalculateDataTransfer(numSamples, lastLayerSize, Ns, sizeOfCiphertext)
		dataTransferTime := CalculateNetworkTime(dataTransferSize, networkRate)

		// Calculate the total estimated time for the server-side operations
		totalEstimatedTime := totalEstimatedRotationTime + dataTransferTime

		// Check if the estimated time is within the desired time
		if totalEstimatedTime <= desiredTime {
			lastSplitIdx = splitIdx
			serverTime = totalEstimatedTime
			serverOperationsDepth = totalDepth
		} else {
			break
		}
	}

	// If no valid split index was found, return the last tested index
	return lastSplitIdx, serverTime, serverOperationsDepth
}

func advisor(rotationTime time.Duration, netArch *NetworkArchitecture, desiredTime time.Duration, numSamples, LogN int, sizeOfCiphertext float64, networkRate float64, degreeOfActivationPolynomial, desiredDepth int, maxMemory float64, maxClientTime time.Duration, params hefloat.Parameters, encoder hefloat.Encoder, encryptor rlwe.Encryptor, evl *bootstrapping.Evaluator) (validSplitIndices []int, estimatedTimes []time.Duration, err error) {
	// Base time
	baseTime := 20 * 60 * time.Second
	// Call the server estimator
	serverSplitIdx, _, _ := serverEstimator(rotationTime, netArch, desiredTime, numSamples, LogN, sizeOfCiphertext, networkRate, degreeOfActivationPolynomial, desiredDepth)

	// Call the client estimator
	clientSplitIdx, _, _ := clientEstimator(netArch, maxMemory, maxClientTime, numSamples)

	// Check if there are no matching split indices
	if clientSplitIdx > serverSplitIdx {
		return nil, nil, fmt.Errorf("no valid split index found, please try another architecture")
	}

	// Check for valid split indices and estimate the total time
	for splitIdx := clientSplitIdx; splitIdx <= serverSplitIdx; splitIdx++ {
		// Set the split index in the network architecture
		netArch.SplitIdx = splitIdx

		//depthOfOperations := desiredDepth
		// Estimate the total time for the given split index
		//encryptedColumns1, encryptedColumns2, _ := InitializeBootstrappingElements(netArch.Layers[splitIdx], numSamples, params, encoder, encryptor)
		//_, timeBootraps := estimateBootstraps(netArch.Layers[splitIdx:], degreeOfActivationPolynomial, depthOfOperations, LogN, evl, encryptedColumns1, encryptedColumns2)

		totalEstimatedTime := estimate(rotationTime, netArch, splitIdx, numSamples, LogN, sizeOfCiphertext, networkRate)

		// Store the valid split index and its corresponding estimated time
		validSplitIndices = append(validSplitIndices, splitIdx)
		estimatedTimes = append(estimatedTimes, totalEstimatedTime+baseTime) //+timeBootraps)
	}

	return validSplitIndices, estimatedTimes, nil
}

func estimateClientMemoryUsage(layers []int, numSamples int) float64 {
	var totalMemory int64
	bytesPerNumber := 4 // Size of float32 in bytes

	// Calculate memory for storing the outputs of each layer
	for _, size := range layers {
		layerMemory := int64(numSamples) * int64(size) * int64(bytesPerNumber)
		totalMemory += layerMemory
	}

	// Convert bytes to megabytes
	totalMemoryUsage := float64(totalMemory) / (1024 * 1024)
	return totalMemoryUsage
}

func estimateBootstraps(serverLayers []int, activationDegree int, maxAllowedDepth int, logN int, evl *bootstrapping.Evaluator, encryptedColumns1 []*rlwe.Ciphertext, encryptedColumns2 []*rlwe.Ciphertext) (int, time.Duration) {
	totalBootstraps := 0
	currentDepth := 0
	totalBootstrapTime := time.Duration(0) // Initialize total bootstrap time
	scalingFactor := 0.0001
	totalHomomorphicAddition := calculateOperations(serverLayers)
	totalHomomorphicMultiplications := totalHomomorphicAddition
	// Bootstraps required for the server layers
	for i := 0; i < len(serverLayers)-1; i++ {
		layerMultiplications := int(float64(serverLayers[i]*serverLayers[i+1])/math.Ceil(math.Pow(2, float64(logN-1)))) + 1
		serverLayerDepth := int(float64(layerMultiplications) * float64(totalHomomorphicMultiplications+totalHomomorphicAddition) * scalingFactor)
		totalLayerDepth := int(float64(serverLayerDepth) + float64(activationDegree*serverLayerDepth)*scalingFactor)
		currentDepth += totalLayerDepth

		// Check if the current depth exceeds the maximum allowed depth
		if currentDepth > maxAllowedDepth {
			// Perform bootstrapping on the ciphertexts and measure time
			bootstrapStart := time.Now()
			for col := 0; col < len(encryptedColumns1); col++ {
				var err error
				encryptedColumns1[col], err = evl.Bootstrap(encryptedColumns1[col])
				if err != nil {
					panic(err)
				}
				encryptedColumns2[col], err = evl.Bootstrap(encryptedColumns2[col])
				if err != nil {
					panic(err)
				}
			}
			bootstrapElapsed := time.Since(bootstrapStart)
			totalBootstrapTime += bootstrapElapsed // Calculate bootstrap time

			totalBootstraps++
			currentDepth = 0 // Reset depth after bootstrapping
		}
	}

	return totalBootstraps, totalBootstrapTime
}
