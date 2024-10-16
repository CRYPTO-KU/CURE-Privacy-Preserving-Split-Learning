package main

import (
	"fmt"
	"math"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type CryptoSystem struct {
	eval      *hefloat.Evaluator
	encoder   *hefloat.Encoder
	encryptor *rlwe.Encryptor
	decyrptor *rlwe.Decryptor
	params    hefloat.Parameters
}

func main() {
	LogN := 14
	max_key_length := 20
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

	kgen := hefloat.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rotations := make([]int, max_key_length)
	for i := 1; i <= max_key_length; i++ {
		rotations[i-1] = int(math.Pow(2, float64(i-1)))
	}

	encoder := hefloat.NewEncoder(params)
	encryptor := hefloat.NewEncryptor(params, pk)
	decryptor := hefloat.NewDecryptor(params, sk)

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

	cryptoSystem := CryptoSystem{
		eval:      eval,
		encoder:   encoder,
		encryptor: encryptor,
		decyrptor: decryptor,
		params:    params,
	}

	// Initialize the matricies

	dataMatrix := InitMatrix(1, 784, 1)
	weightMatrix := InitMatrix(784, 128, 1)
	resultMatrix := InitMatrix(1, 128, 1)
	fmt.Println("working")

	resultMatrix.Mul(dataMatrix, weightMatrix)

	start := time.Now()
	he_result := CCmatrixmatrixDotParalellized(dataMatrix, weightMatrix, cryptoSystem)
	end := time.Since(start)

	fmt.Print(he_result.At(0, 0))
	fmt.Printf("\nElapsed Time in Matrix Matrix Dot product: %v", end)

}

func matPrint(m mat.Matrix) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf("%6.2f ", m.At(i, j))
		}
		fmt.Println()
	}
}

func PrintVector(vector []complex128) {
	fmt.Print("[")
	for i, v := range vector {
		fmt.Printf("%.2f%+.2fi", real(v), imag(v))
		if i < len(vector)-1 {
			fmt.Print(" ")
		}
	}
	fmt.Println("]")
}

func inner_product(ct1, ct2 *rlwe.Ciphertext, length_of_ct int, CryptoSystem CryptoSystem, P int) (result []float64, err error) {

	ct, _ := CryptoSystem.eval.MulNew(ct1, ct2)
	CryptoSystem.eval.Relinearize(ct, ct)
	var vect1 []complex128
	pt1 := hefloat.NewPlaintext(CryptoSystem.params, CryptoSystem.params.MaxLevel())
	if err := CryptoSystem.encoder.Encode(vect1, pt1); err != nil {
		panic(err)
	}

	CryptoSystem.encoder.Encode(vect1, pt1)
	ct_result, err := CryptoSystem.encryptor.EncryptNew(pt1)

	CryptoSystem.eval.Add(ct, ct_result, ct_result)

	for i := 1; i < length_of_ct; i = i * 2 {
		adder, _ := CryptoSystem.eval.RotateNew(ct_result, i)
		CryptoSystem.eval.Add(ct_result, adder, ct_result)
	}
	result_product := mask_first(ct_result, CryptoSystem, P)
	return result_product, nil
}

func CCmatrixmatrixDot(mat1, mat2 *mat.Dense, cryptoSystem CryptoSystem) (result *mat.Dense) {
	rows1, cols1 := mat1.Dims()
	rows2, cols2 := mat2.Dims()

	mat1 = MatrixPadder(mat1)
	mat2 = MatrixPadder(mat2)

	P := 16384 //numSlots

	if cols1 != rows2 {
		var err error
		panic(err)
	}

	result = mat.NewDense(rows1, cols2, nil)
	for i := 0; i < rows1; i++ {
		for j := 0; j < cols2; j++ {
			var sum []float64
			vect1 := GetRowAsFloat(mat1, i)
			vect2 := GetColumnAsFloat(mat2, j)

			pt1 := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
			if err := cryptoSystem.encoder.Encode(vect1, pt1); err != nil {
				panic(err)
			}

			pt2 := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
			if err := cryptoSystem.encoder.Encode(vect2, pt2); err != nil {
				panic(err)
			}
			cryptoSystem.encoder.Encode(vect1, pt1)
			cryptoSystem.encoder.Encode(vect2, pt2)

			ct1, _ := cryptoSystem.encryptor.EncryptNew(pt1)
			ct2, _ := cryptoSystem.encryptor.EncryptNew(pt2)

			sum, _ = inner_product(ct1, ct2, cols1, cryptoSystem, P)
			for k := 0; k < len(sum); k++ {
				result.Set(i+k, j, sum[k])
			}
		}
	}
	result = Submatrix(result, rows1, cols2)
	return result
}

// isPowerOfTwo checks if a number is a power of 2
func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

func SmallestPowerOfTwo(n int) int {
	if n <= 0 {
		return 1 // Return 1 for non-positive numbers
	}
	return int(math.Pow(2, math.Ceil(math.Log2(float64(n)))))
}

func GetColumnAsFloat(matrix *mat.Dense, colIndex int) []float64 {
	rows, _ := matrix.Dims()
	column := make([]float64, rows)
	for i := 0; i < rows; i++ {
		column[i] = matrix.At(i, colIndex)
	}
	return column
}

func GetRowAsFloat(matrix *mat.Dense, rowIndex int) []float64 {
	_, cols := matrix.Dims()
	row := make([]float64, cols)
	for j := 0; j < cols; j++ {
		row[j] = matrix.At(rowIndex, j)
	}
	return row
}

func mask_first(ct *rlwe.Ciphertext, cryptoSystem CryptoSystem, P int) []float64 {
	vect1 := []complex128{0 + 0i}
	pt := cryptoSystem.decyrptor.DecryptNew(ct)
	cryptoSystem.encoder.Decode(pt, vect1)
	vectResult := maskValues(vect1, P)
	return vectResult
}

func MatrixPadder(matrix *mat.Dense) *mat.Dense {
	rows, cols := matrix.Dims()

	// Find the nearest power of 2 for the number of rows and columns
	paddedRows := int(math.Pow(2, math.Ceil(math.Log2(float64(rows)))))
	paddedCols := int(math.Pow(2, math.Ceil(math.Log2(float64(cols)))))

	// Create a new matrix with the padded dimensions
	paddedMatrix := mat.NewDense(paddedRows, paddedCols, nil)

	// Copy the original matrix into the padded matrix
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			paddedMatrix.Set(i, j, matrix.At(i, j))
		}
	}

	return paddedMatrix
}

func Submatrix(matrix *mat.Dense, rows, cols int) *mat.Dense {
	// Get the dimensions of the original matrix
	origRows, origCols := matrix.Dims()

	// Check if the requested rows and columns are within the bounds of the original matrix
	if rows > origRows || cols > origCols {
		panic("Requested submatrix dimensions exceed the dimensions of the original matrix")
	}

	// Extract the submatrix
	submatrix := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			submatrix.Set(i, j, matrix.At(i, j))
		}
	}

	return submatrix
}

func InitMatrix(rows, cols int, value float64) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = value
	}
	return mat.NewDense(rows, cols, data)
}

func CCmatrixmatrixDotParalellized(mat1, mat2 *mat.Dense, cryptoSystem CryptoSystem) *mat.Dense {
	rows1, cols1 := mat1.Dims()
	rows2, cols2 := mat2.Dims()

	P := 1024 //number of slots of an rlwe interface of the encryption scheme

	mat1 = MatrixPadder(mat1)
	mat2 = MatrixPadder(mat2)

	padded_rows1, padded_cols1 := mat1.Dims()
	padded_rows2, padded_cols2 := mat2.Dims()

	if cols1 != rows2 {
		var err error
		panic(err)
	}

	batched_mat1 := reshapeMatrix(mat1, padded_rows1*padded_cols1/P, P)
	batched_mat2 := expandMatrix(mat2, P/padded_rows2)

	batched_rows1, _ := batched_mat1.Dims()
	//_, batched_cols2 := batched_mat2.Dims()

	var wg sync.WaitGroup
	wg.Add(batched_rows1)

	var data []float64

	for i := 0; i < batched_rows1; i++ {
		go func(i int) {
			defer wg.Done()
			for j := 0; j < cols2; j++ {
				var sum []float64
				vect1 := GetRowAsFloat(batched_mat1, i)
				vect2 := GetColumnAsFloat(batched_mat2, j)

				pt1 := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
				if err := cryptoSystem.encoder.Encode(vect1, pt1); err != nil {
					panic(err)
				}

				pt2 := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
				if err := cryptoSystem.encoder.Encode(vect2, pt2); err != nil {
					panic(err)
				}

				ct1, _ := cryptoSystem.encryptor.EncryptNew(pt1)
				ct2, _ := cryptoSystem.encryptor.EncryptNew(pt2)

				sum, _ = inner_product(ct1, ct2, cols1, cryptoSystem, P)
				data = append(data, sum...)
			}
		}(i)
	}

	wg.Wait()
	result := mat.NewDense(padded_rows1, padded_cols2, data)
	result = Submatrix(result, rows1, cols2)

	return result
}

func reshapeMatrix(src *mat.Dense, newRows, newCols int) *mat.Dense {
	r, c := src.Dims()

	// Check if the total elements match
	if r*c != newRows*newCols {
		panic("new dimensions do not match the number of elements in the original matrix")
	}

	// Create a new matrix with the desired shape
	data := src.RawMatrix().Data
	dst := mat.NewDense(newRows, newCols, data)

	return dst
}

func expandMatrix(src *mat.Dense, repeats int) *mat.Dense {
	if repeats < 1 {
		panic("repeats must be at least 1")
	}

	r, c := src.Dims()
	newRows := r * repeats
	newData := make([]float64, newRows*c)

	// Fill new data by repeating the original matrix rows
	for i := 0; i < repeats; i++ {
		for j := 0; j < r; j++ {
			copy(newData[(i*r+j)*c:(i*r+j+1)*c], src.RawRowView(j))
		}
	}

	// Create a new Dense matrix with the repeated rows
	dst := mat.NewDense(newRows, c, newData)

	return dst
}

func maskValues(data []complex128, numSlots int) []float64 {
	var result []float64
	for i := 0; i < len(data); i += numSlots {
		result = append(result, real(data[i]))
	}
	return result
}
