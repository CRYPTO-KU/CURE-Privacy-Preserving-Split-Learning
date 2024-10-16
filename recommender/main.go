package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"

	//"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"gonum.org/v1/gonum/mat"
)

type CryptoSystem struct {
	//eval      *hefloat.Evaluator
	encoder   *hefloat.Encoder
	encryptor *rlwe.Encryptor
	//decyrptor *rlwe.Decryptor
	params hefloat.Parameters
}

type NetworkArchitecture struct {
	Layers   []int // Full sequence of layer sizes
	SplitIdx int   // Index in Layers where the split occurs (index 'c')
}

func main() {

	var num_CPU int
	//var int_batch_size int
	flag.IntVar(&num_CPU, "cpu", runtime.NumCPU(), "number of CPUs to use")
	flag.Parse()

	// Limit the number of CPUs to use
	runtime.GOMAXPROCS(num_CPU)
	fmt.Printf("Number of CPUs used: %d\n", num_CPU)

	// Setting up the parameters for HE
	LogN := 16 // For example, N=2^LogN slots
	max_key_length := 20
	/*
		params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
			LogN: LogN,
			Q: []uint64{0x200000008001, 0x400018001, // 45 + 9 x 34
				0x3fffd0001, 0x400060001,
				0x400068001, 0x3fff90001,
				0x400080001, 0x4000a8001,
				0x400108001, 0x3ffeb8001},
			P:               []uint64{0x7fffffd8001, 0x7fffffc8001}, // 43, 43
			LogDefaultScale: 40,                                     // Log2 of the scale
		})
		if err != nil {
			panic(err)
		}
	*/
	params, err := hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN:            LogN,
			LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
			LogP:            []int{61, 61, 61},
			LogDefaultScale: 40,
		})
	if err != nil {
		panic(err)
	}

	// Key generation
	kgen := hefloat.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rotations := make([]int, max_key_length)
	for i := 1; i <= max_key_length; i++ {
		rotations[i-1] = int(math.Pow(2, float64(i-1)))
	}

	encoder := hefloat.NewEncoder(params)
	encryptor := hefloat.NewEncryptor(params, pk)
	//decryptor := hefloat.NewDecryptor(params, sk)

	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := hefloat.NewEvaluator(params, evk)

	galEls := []uint64{
		params.GaloisElement(1),
		params.GaloisElement(2),
		params.GaloisElement(4),
		params.GaloisElement(8),
		params.GaloisElement(16),
		params.GaloisElement(32),
		params.GaloisElement(64),
		params.GaloisElement(128),
		params.GaloisElement(256),
		params.GaloisElement(512),
		params.GaloisElement(1024),
		params.GaloisElement(2048),
		params.GaloisElement(4096),
		params.GaloisElement(8192),
	}
	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

	/*btpParametersLit := bootstrapping.ParametersLiteral{
		LogP: []int{61, 61, 61, 61},
		Xs:   params.Xs(),
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		panic(err)
	}
	//fmt.Println("Generating bootstrapping evaluation keys...")
	evak, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(err)
	}
	var evl *bootstrapping.Evaluator
	if evl, err = bootstrapping.NewEvaluator(btpParams, evak); err != nil {
		panic(err)
	}*/

	cryptoSystem := CryptoSystem{
		//eval:      eval,
		encoder:   encoder,
		encryptor: encryptor,
		//decyrptor: decryptor,
		params: params,
	}

	//end of key generations

	//numCols := int_batch_size
	//numRows := 8192
	//totalBootstraps := 0
	//Bootstraps := 0
	numSamples := 10000                  // number of data samples
	totalHomomorphicMultiplications := 0 // Initialize the counter for homomorphic multiplications
	totalHomomorphicAddition := 0        // Initialize the counter for homomorphic addition
	layers := []int{784, 128, 32, 10}
	splitIdx := 2
	//serverLayers := layers[:splitIdx]
	//clientLayers := layers[splitIdx:]
	netArch := NewNetworkArchitecture(layers, splitIdx)
	desiredTrainingTime := 10000 * time.Second
	sizeOfCiphertext := 0.0078125
	networkRate := 1.0 // Network rate in MBps
	/*var vector_batches []int
	for i := 0; i < len(serverLayers)-1; i++ {
		vector_batches = append(vector_batches, int(float64(serverLayers[i]*serverLayers[i+1])/math.Ceil(math.Pow(2, float64(LogN-1))))+1)
	}

	var total_iterations []int
	for i := 0; i < len(serverLayers)-1; i++ {
		total_iterations = append(total_iterations, vector_batches[i]*numSamples/int_batch_size)
	}

	total_iterations = append(total_iterations, int(math.Ceil(float64(serverLayers[len(serverLayers)-1]*clientLayers[0]*numSamples)/(float64(int_batch_size)*math.Ceil(math.Pow(2, float64(LogN-1)))))))

	encryptedColumns1 := make([]*rlwe.Ciphertext, numCols)
	encryptedColumns2 := make([]*rlwe.Ciphertext, numCols)
	plainColumns := make([]*rlwe.Plaintext, numCols)

	matrix1 := generateMatrix(numRows, numCols)
	matrix2 := generateMatrix(numRows, numCols)

	for col := 0; col < numCols; col++ {
		// Extract the current column from matrix1 and matrix2
		column1 := make([]float64, len(matrix1))
		column2 := make([]float64, len(matrix2))
		for row := 0; row < len(matrix1); row++ {
			column1[row] = matrix1[row][col]
			column2[row] = matrix2[row][col]
		}

		// Encode the current column
		pt1 := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(column1, pt1); err != nil {
			panic(err)
		}

		pt2 := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(column2, pt2); err != nil {
			panic(err)
		}

		Plaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(column2, pt2); err != nil {
			panic(err)
		}

		// Encrypt the encoded columns
		encryptedColumn1, err := encryptor.EncryptNew(pt1)
		if err != nil {
			panic(err)
		}

		encryptedColumn2, err := encryptor.EncryptNew(pt2)
		if err != nil {
			panic(err)
		}
		bootstrappedColumn1, err := evl.Bootstrap(encryptedColumn1)
		if err != nil {
			panic(err)
		}
		Bootstraps++
		bootstrappedColumn2, err := evl.Bootstrap(encryptedColumn2)
		if err != nil {
			panic(err)
		}
		totalBootstraps++
		encryptedColumns1[col] = bootstrappedColumn1
		encryptedColumns2[col] = bootstrappedColumn2
		plainColumns[col] = Plaintext1
	}

	for i := 0; i < len(total_iterations)-2; i++ {
		totalHomomorphicMultiplications += performCCParallelMultiplication(encryptedColumns1, encryptedColumns2, *eval, max_key_length) * total_iterations[i]
		totalHomomorphicAddition += performCCParallelAddition(encryptedColumns1, encryptedColumns2, *eval, max_key_length) * total_iterations[i]
	}

	totalHomomorphicMultiplications += performCPParallelMultiplication(encryptedColumns1, plainColumns, *eval, max_key_length) * total_iterations[len(total_iterations)-1]
	totalHomomorphicAddition += performCPParallelAddition(encryptedColumns1, plainColumns, *eval, max_key_length) * total_iterations[len(total_iterations)-1]*/

	var vect1 []complex128
	var cts []*rlwe.Ciphertext
	batch_size := 120
	for i := 0; i < batch_size; i++ {
		pt := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
		if err := cryptoSystem.encoder.Encode(vect1, pt); err != nil {
			panic(err)
		}
		ct, _ := cryptoSystem.encryptor.EncryptNew(pt)
		cts = append(cts, ct)
	}

	// Variable to store average times
	var wg sync.WaitGroup

	// Specify the total number of rotations to execute in parallel
	start := time.Now()

	// Assuming total_rotation_executions applies to each ciphertext
	wg.Add(len(cts) * num_CPU) // Make sure this matches the exact number of goroutines you start
	for _, ct := range cts {
		for j := 0; j < num_CPU; j++ { // This should match the second part of wg.Add
			go func(ct *rlwe.Ciphertext, j int) {
				defer wg.Done()
				eval.RotateNew(ct, 1)
			}(ct, j)
		}
	}

	wg.Wait()
	totalDuration := time.Since(start)
	// Calculate the average duration for the rotation by 1
	averageDuration := totalDuration / time.Duration(len(cts)*num_CPU)
	rotationTime := averageDuration // Example rotation time
	degreeOfActivationPolynomial := 3
	desiredDepth := 15

	Estimator(rotationTime, netArch, desiredTrainingTime, numSamples, totalHomomorphicMultiplications, totalHomomorphicAddition, LogN, sizeOfCiphertext, networkRate)
	//------------------------------------------------------------------------------------------------------------//
	//-------------------------------------------Client Estimator-------------------------------------------------//
	//------------------------------------------------------------------------------------------------------------//
	maxMemory := 500.0 // MB
	maxClientTime := 2 * time.Second

	lastSplitIdx, clientTime, clientMemoryUsage := clientEstimator(netArch, maxMemory, maxClientTime, numSamples)

	fmt.Println("----- Client Estimator Part -----")
	// Output the results
	fmt.Println("Last successful split index:", lastSplitIdx)
	fmt.Println("Client time taken:", clientTime)
	fmt.Printf("Client memory usage: %.2f MB\n", clientMemoryUsage)

	//------------------------------------------------------------------------------------------------------------//
	//-------------------------------------------Server Estimator-------------------------------------------------//
	//------------------------------------------------------------------------------------------------------------//

	lastSplitIdx, serverTime, serverOperationsDepth := serverEstimator(rotationTime, netArch, desiredTrainingTime, numSamples, totalHomomorphicMultiplications, totalHomomorphicAddition, LogN, sizeOfCiphertext, networkRate, degreeOfActivationPolynomial, desiredDepth)

	fmt.Println("----- Server Estimator Part -----")
	fmt.Println("Last successful split index:", lastSplitIdx)
	fmt.Println("Server time taken:", serverTime)
	fmt.Printf("Server operations depth: %.2f\n", serverOperationsDepth)
}

func NewNetworkArchitecture(layers []int, splitIdx int) *NetworkArchitecture {
	return &NetworkArchitecture{
		Layers:   layers,
		SplitIdx: splitIdx,
	}
}

// Calculate the smallest power of two greater than x
func SmallestPowerOfTwo(x int) int {
	return int(math.Pow(2, math.Ceil(math.Log2(float64(x)))))
}

// Calculate the data transfer size based on the network architecture and data samples
func CalculateDataTransfer(numSamples int, lastLayerSize int, numSlots int, sizeOfCiphertext float64) float64 {
	return float64(numSamples*lastLayerSize) / float64(numSlots/2) * sizeOfCiphertext
}

// Calculate network data transfer time
func CalculateNetworkTime(dataTransferSize float64, networkRate float64) time.Duration {
	// Assuming network rate is in MB per second
	return time.Duration((dataTransferSize / networkRate) * float64(time.Second))
}

func generateRandomDenseMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()
	}
	return mat.NewDense(rows, cols, data)
}

func microBenchmarkDotProduct(clientLayers []int, numSamples int) time.Duration {
	var totalTime time.Duration
	const iterations = 10

	for i := 1; i < len(clientLayers)-1; i++ {
		sizeLiMinus1 := clientLayers[i-1]
		sizeLi := clientLayers[i]
		sizeLiPlus1 := clientLayers[i+1]

		// Create matrices for dot product using gonum
		matrixA := generateRandomDenseMatrix(sizeLiMinus1, sizeLi)
		matrixB := generateRandomDenseMatrix(sizeLi, sizeLiPlus1)

		// Measure dot product time over 10 iterations
		for j := 0; j < iterations; j++ {
			start := time.Now()
			var result mat.Dense
			result.Mul(matrixA, matrixB)
			totalTime += time.Since(start)
		}
	}

	// Return the average time for the dot product
	return totalTime / iterations
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

func createMatrices(array []int) ([]*mat.Dense, error) {
	if len(array) < 2 {
		return nil, fmt.Errorf("array should have at least two elements")
	}

	matrices := make([][][]int, len(array)-1)

	for i := 0; i < len(array)-1; i++ {
		rows := array[i]
		cols := array[i+1]
		matrix := make([][]int, rows)
		for j := range matrix {
			matrix[j] = make([]int, cols)
		}
		matrices[i] = matrix
	}

	var denseMatrices []*mat.Dense

	for i := 0; i < len(matrices); i++ {
		toDense(matrices[i])
	}

	return denseMatrices, nil
}

func toDense(matrix [][]int) *mat.Dense {
	rows := len(matrix)
	cols := len(matrix[0])
	data := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = float64(matrix[i][j])
		}
	}
	return mat.NewDense(rows, cols, data)
}

func generateMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64()
		}
	}
	return matrix
}

// Cipher text, cipher text homomorphic dot product
func performCCParallelMultiplication(encryptedColumns1, encryptedColumns2 []*rlwe.Ciphertext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	multiplications := 0 // Counter for homomorphic multiplications

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			for i := 1; i < max_key+1; i = i * 2 {
				resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
				eval.Rotate(encryptedColumns1[col], i, encryptedColumns1[col])
				multiplications++ // Increment the counter for each multiplication
			}
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}

	return multiplications
}

// Cipher text, cipher text homomorphic addition
func performCCParallelAddition(encryptedColumns1, encryptedColumns2 []*rlwe.Ciphertext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	addition := 0 // Counter for homomorphic addition

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
			addition++ // Increment the counter for each addition
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}

	return addition
}

// Cipher text, plain text homomorphic dot product
func performCPParallelMultiplication(encryptedColumns1 []*rlwe.Ciphertext, encryptedColumns2 []*rlwe.Plaintext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	multiplications := 0 // Counter for homomorphic multiplications

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			for i := 1; i < max_key+1; i = i * 2 {
				resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
				eval.Rotate(encryptedColumns1[col], i, encryptedColumns1[col])
				multiplications++ // Increment the counter for each multiplication
			}
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}

	return multiplications
}

// Cipher text, plain text homomorphic addition
func performCPParallelAddition(encryptedColumns1 []*rlwe.Ciphertext, encryptedColumns2 []*rlwe.Plaintext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	addition := 0 // Counter for homomorphic addition

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
			addition++ // Increment the counter for each addition
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}

	return addition
}

func estimateBootstraps(serverLayers, clientLayers []int, totalHomomorphicMultiplications int, totalHomomorphicAddition int, logN int) int {
	totalBootstraps := 0
	TB := 0
	ServerBT := 0
	// Calculate the bootstraps required for the server layers
	for i := 0; i < len(serverLayers)-1; i++ {
		layerMultiplications := int(float64(serverLayers[i]*serverLayers[i+1])/math.Ceil(math.Pow(2, float64(logN-1)))) + 1
		ServerBT += layerMultiplications * totalHomomorphicMultiplications * totalHomomorphicAddition
	}

	// Calculate the bootstraps required for the transition from the last server layer to the first client layer
	lastServerLayer := serverLayers[len(serverLayers)-1]
	firstClientLayer := clientLayers[0]
	clientMultiplications := int(math.Ceil(float64(lastServerLayer*firstClientLayer*totalHomomorphicMultiplications*totalHomomorphicAddition) / (math.Ceil(math.Pow(2, float64(logN-1))))))
	TB += clientMultiplications
	totalBootstraps = TB + ServerBT

	return totalBootstraps
}

// Estimator function to calculate rotations and suggest network properties
func Estimator(rotationTime time.Duration, netArch *NetworkArchitecture, desiredTime time.Duration, numSamples, totalHomomorphicMultiplications, totalHomomorphicAddition, LogN int, sizeOfCiphertext float64, networkRate float64) {
	serverLayers := netArch.Layers[:netArch.SplitIdx]
	clientLayers := netArch.Layers[netArch.SplitIdx:]

	Ns := 8192 // LogN Number of slots
	var totalRotations int

	for i := 0; i < len(serverLayers)-1; i++ {
		sizeLi := SmallestPowerOfTwo(serverLayers[i])
		sizeLiPlus1 := SmallestPowerOfTwo(serverLayers[i+1])
		rotationsForLayer := numSamples * sizeLi * sizeLiPlus1 / Ns * int(math.Log2(float64(sizeLiPlus1)))
		totalRotations += rotationsForLayer
	}

	totalEstimatedRotationTime := time.Duration(totalRotations) * rotationTime

	lastLayerSize := serverLayers[len(serverLayers)-1]
	dataTransferSize := CalculateDataTransfer(numSamples, lastLayerSize, Ns, sizeOfCiphertext)
	dataTransferTime := CalculateNetworkTime(dataTransferSize, networkRate)
	clientMemoryUsage := estimateClientMemoryUsage(clientLayers, numSamples)

	totalEstimatedTime := totalEstimatedRotationTime + dataTransferTime

	totalEstimatedBootstraps := estimateBootstraps(serverLayers, clientLayers, totalHomomorphicMultiplications, totalHomomorphicAddition, LogN)

	fmt.Printf("Rotation Time per Layer: %v\n", rotationTime)
	fmt.Printf("Estimated Total Time for all Rotations: %v\n", totalEstimatedRotationTime)
	fmt.Printf("Estimated Data Transfer Time: %v\n", dataTransferTime)
	fmt.Printf("Total Estimated Time: %v\n", totalEstimatedTime)
	fmt.Println("Server Layers:", serverLayers)
	fmt.Println("Client Layers:", clientLayers)
	fmt.Println("Total number of bootstraps estimated", totalEstimatedBootstraps)
	fmt.Println("Client total memory usage (MB):", clientMemoryUsage)
	fmt.Println("Total Rotations Needed:", totalRotations)
	fmt.Printf("Data Transfer for Last Layer (MB): %.6f\n", dataTransferSize)

	if totalEstimatedTime <= desiredTime {
		fmt.Println("The desired training time can be achieved.")
	} else {
		fmt.Println("The desired training time cannot be achieved, please adjust the network architecture or improve rotation speed.")
	}
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

func serverEstimator(rotationTime time.Duration, netArch *NetworkArchitecture, desiredTime time.Duration, numSamples, totalHomomorphicMultiplications, totalHomomorphicAddition, LogN int, sizeOfCiphertext float64, networkRate float64, degreeOfActivationPolynomial int, desiredDepth int) (lastSplitIdx int, serverTime time.Duration, serverOperationsDepth float64) {
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
			rotationsForLayer := numSamples * sizeLi * sizeLiPlus1 / Ns * int(math.Log2(float64(sizeLiPlus1)))
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
