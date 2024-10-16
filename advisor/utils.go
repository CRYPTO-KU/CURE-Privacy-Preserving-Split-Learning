package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"gonum.org/v1/gonum/mat"
)

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
func measureRotationTime(cts []*rlwe.Ciphertext, wg *sync.WaitGroup, num_CPU int, eval hefloat.Evaluator) time.Duration {
	start := time.Now()

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
	return time.Since(start) / time.Duration(len(cts)*num_CPU)
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

func calculateOperations(layers []int) int {
	totalOperations := 0

	// Adjust all layers to the smallest power of two greater than or equal to their size
	for i := range layers {
		layers[i] = SmallestPowerOfTwo(layers[i])
	}

	// Sum the logarithms of the adjusted layer sizes from l2 to ln
	for i := 1; i < len(layers); i++ {
		totalOperations += int(math.Log2(float64(layers[i])))
	}

	return totalOperations
}

func InitializeBootstrappingElements(numRows, numCols int, params hefloat.Parameters, encoder hefloat.Encoder, encryptor rlwe.Encryptor) ([]*rlwe.Ciphertext, []*rlwe.Ciphertext, []*rlwe.Plaintext) {
	matrix1 := generateMatrix(numRows, numCols)
	matrix2 := generateMatrix(numRows, numCols)

	encryptedColumns1 := make([]*rlwe.Ciphertext, numCols)
	encryptedColumns2 := make([]*rlwe.Ciphertext, numCols)
	plainColumns := make([]*rlwe.Plaintext, numCols)

	totalBootstraps := 0 // Initialize the counter for bootstrap operations

	for col := 0; col < numCols; col++ {
		column1 := make([]float64, len(matrix1))
		column2 := make([]float64, len(matrix2))
		for row := 0; row < len(matrix1); row++ {
			column1[row] = matrix1[row][col]
			column2[row] = matrix2[row][col]
		}

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

		encryptedColumn1, err := encryptor.EncryptNew(pt1)
		if err != nil {
			panic(err)
		}

		encryptedColumn2, err := encryptor.EncryptNew(pt2)
		if err != nil {
			panic(err)
		}

		totalBootstraps++
		encryptedColumns1[col] = encryptedColumn1
		encryptedColumns2[col] = encryptedColumn2
		plainColumns[col] = Plaintext1
	}

	return encryptedColumns1, encryptedColumns2, plainColumns
}

func parseArchitecture(archStr string) []int {
	// Split the string by commas
	archParts := strings.Split(archStr, ",")
	arch := make([]int, len(archParts))

	// Convert each part from string to integer
	for i, part := range archParts {
		val, err := strconv.Atoi(part)
		if err != nil {
			fmt.Printf("Error parsing architecture: %v\n", err)
			return nil
		}
		arch[i] = val
	}
	return arch
}
