package m

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	// "github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func ckksDotVectorVector(v1, v2 *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	evaluator.MulRelinNew(v1, v2)
	v1.Copy(v2)
	levelFloat := float64(v1.Level())
	level := math.Log(levelFloat)
	for i := level; i > 1; i = i / 2 {
		j := int(i)
		evaluator.RotateNew(v1, j)
		evaluator.AddNew(v1, v2)
	}
	return v2
}

func ckksDotMatrixVector(A *rlwe.Ciphertext, vector mat.Matrix, evaluator hefloat.Evaluator, m, n int, params hefloat.Parameters, encoder hefloat.Encoder) *rlwe.Ciphertext {

	v := make([]float64, m)
	for i := 0; i < m; i++ {
		v[i] = vector.At(i, 0)
	}

	result := make([]float64, len(v)*m)
	for i := 0; i < m; i++ {
		copy(result[i*len(v):(i+1)*len(v)], v)
	}
	pt1 := hefloat.NewPlaintext(params, params.MaxLevel())
	w := encoder.Encode(result, pt1)

	evaluator.MulRelinNew(A, w)
	B := A.CopyNew()
	for i := 1; i < n; i++ {
		evaluator.RotateNew(B, -i*m)
		evaluator.AddNew(A, B)
	}
	return A
}

func ckksAdd(m, n *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	evaluator.Add(m, n, m)
	return m
}

func ckksMultiply(m, n *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	evaluator.Mul(m, n, m)
	return m
}
func MultByConst(ctIn *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) {
}

func ckksScale(m *rlwe.Ciphertext, s float64, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	MultByConst(m, s, m)
	return m
}
func AddConst(ctIn *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) {
}

func ckksAddScalar(m *rlwe.Ciphertext, s float64, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	AddConst(m, s, m)
	return m
}

func ckksSubtract(m, n *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	evaluator.Sub(m, n, m)
	return m
}

func multiClassCrossEntropyLoss(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, CalculateSignedLogarithmMatrix(n))
	return o
}

func CalculateSignedLogarithmMatrix(input mat.Matrix) *mat.Dense {
	rows, cols := input.Dims()
	logMatrix := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			logMatrix.Set(i, j, -math.Log(val))
		}
	}
	return logMatrix
}

func ApplySoftmaxToColumns(matrix mat.Matrix) (*mat.Dense, error) {
	r, c := matrix.Dims()
	softmaxedMatrix := mat.NewDense(r, c, nil)

	for j := 0; j < c; j++ {
		column := getColumn(matrix, j)
		softmaxedColumn := ApplySoftmaxToVector(column)

		if err := setColumn(softmaxedMatrix, j, softmaxedColumn); err != nil {
			return nil, err
		}
	}

	return softmaxedMatrix, nil
}

func getColumn(matrix mat.Matrix, colIdx int) []float64 {
	r, _ := matrix.Dims()
	column := make([]float64, r)
	for i := 0; i < r; i++ {
		column[i] = matrix.At(i, colIdx)
	}
	return column
}

func setColumn(matrix *mat.Dense, colIdx int, column []float64) error {
	r, _ := matrix.Dims()
	if len(column) != r {
		return fmt.Errorf("column length doesn't match matrix rows")
	}
	for i := 0; i < r; i++ {
		matrix.Set(i, colIdx, column[i])
	}
	return nil
}

func ApplySoftmaxToVector(inputVector []float64) []float64 {
	softmaxed := make([]float64, len(inputVector))
	sum := 0.0
	for i := range inputVector {
		sum += math.Exp(inputVector[i])
	}

	for i := range softmaxed {
		softmaxed[i] = math.Exp(inputVector[i]) / sum
	}

	return softmaxed
}

func randomArray(size int, v float64) []float64 {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return data
}

func addBiasNodeTo(m mat.Matrix, b float64) mat.Matrix {
	r, _ := m.Dims()
	a := mat.NewDense(r+1, 1, nil)

	a.Set(0, 0, b)
	for i := 0; i < r; i++ {
		a.Set(i+1, 0, m.At(i, 0))
	}
	return a
}

func toMatrix(v []float64, m, n int) [][]float64 {
	matrix := make([][]float64, m)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = v[i*n+j]
		}
	}

	return matrix
}

func toVector(matrix *mat.Dense) []float64 {
	m, n := matrix.Dims()

	vector := make([]float64, m*n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			vector[j*m+i] = matrix.At(i, j)
		}
	}

	return vector
}

func MatrixToVector(matrix mat.Matrix) []float64 {
	r, c := matrix.Dims()
	vector := make([]float64, r*c)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			vector[i*c+j] = matrix.At(i, j)
		}
	}

	return vector
}

func VectorToMatrix(v []float64, m, n int) mat.Matrix {
	matrix := mat.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			matrix.Set(i, j, v[i*n+j])
		}
	}

	return matrix
}

func extendVector(vector, extensionVector []float64) ([]float64, error) {
	len1 := len(vector)
	len2 := len(extensionVector)

	if len2 == 0 {
		return nil, fmt.Errorf("Extension vector is empty")
	}

	// Calculate the number of times extensionVector should be repeated
	repeat := len1 / len2

	// Initialize the extended vector
	extended := make([]float64, len1)

	// Extend the extensionVector to the target vector
	for i := 0; i < len2; i++ {
		for j := 0; j < repeat; j++ {
			extended[j+i*repeat] = extensionVector[i]
		}
	}

	return extended, nil
}

func OneLevelScalarMultiThread(encoder *hefloat.Encoder, encryptor *rlwe.Encryptor, evaluator hefloat.Evaluator, decryptor *rlwe.Decryptor, params hefloat.Parameters, matrix1, matrix2 *mat.Dense) []float64 {
	rowNumMatrix, _ := matrix1.Dims()

	vector1 := toVector(matrix1)
	vector2 := toVector(matrix2)
	vector2, _ = extendVector(vector1, vector2)

	vectors1, colnumber1, lastnumber1 := splitData(vector1, rowNumMatrix, params.LogMaxSlots())
	vectors2, _, _ := splitData(vector2, rowNumMatrix, params.LogMaxSlots())

	splitAmount := len(vectors1)
	encodedVectors1 := make([]*rlwe.Plaintext, splitAmount)
	encodedVectors2 := make([]*rlwe.Plaintext, splitAmount)

	for i := 0; i < splitAmount; i++ {
		encodedVectors1[i] = hefloat.NewPlaintext(params, params.MaxLevel())
		encodedVectors2[i] = hefloat.NewPlaintext(params, params.MaxLevel())

		if err := encoder.Encode(vectors1[i], encodedVectors1[i]); err != nil {
			panic(err)
		}

		if err := encoder.Encode(vectors2[i], encodedVectors2[i]); err != nil {
			panic(err)
		}
	}

	encryptedVectors1 := make([]*rlwe.Ciphertext, splitAmount)
	initials := make([]*rlwe.Ciphertext, splitAmount)

	for i := 0; i < splitAmount; i++ {
		ciphertext, err := encryptor.EncryptNew(encodedVectors1[i])
		if err != nil {
			panic(err)
		}
		encryptedVectors1[i] = ciphertext
	}

	for i := 0; i < splitAmount; i++ {
		evaluator.Mul(encryptedVectors1[i], encodedVectors2[i], encryptedVectors1[i])
		initials[i] = encryptedVectors1[i].CopyNew()
	}

	for i := 0; i < len(encryptedVectors1); i++ {

		if i != len(encryptedVectors1)-1 {
			for j := 1; j < colnumber1; j++ {
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i])
			}
		} else {
			for j := 1; j < lastnumber1; j++ {
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i])
			}
		}

	}

	for i := 1; i < len(encryptedVectors1); i++ {
		evaluator.Add(encryptedVectors1[0], encryptedVectors1[i], encryptedVectors1[0])
	}

	resultPlaintext := decryptor.DecryptNew(encryptedVectors1[0])
	resultComplex := make([]complex128, resultPlaintext.Slots())
	if err := encoder.Decode(resultPlaintext, resultComplex); err != nil {
		panic(err)
	}

	resultFloat := make([]float64, len(resultComplex))
	for i, v := range resultComplex {
		resultFloat[i] = real(v)
	}

	return resultFloat[:rowNumMatrix]
}

func OneLevelHEMultiThread(encoder *hefloat.Encoder, evaluator hefloat.Evaluator, decryptor *rlwe.Decryptor, params hefloat.Parameters, matrix1, matrix2 *mat.Dense) []float64 {
	rowNumMatrix, _ := matrix1.Dims()

	vector1 := toVector(matrix1)
	vector2 := toVector(matrix2)
	vector2, _ = extendVector(vector1, vector2)

	vectors1, colnumber1, lastnumber1 := splitData(vector1, rowNumMatrix, 1024)
	vectors2, _, _ := splitData(vector2, rowNumMatrix, 1024)

	splitAmount := len(vectors1)
	encodedVectors1 := make([]*rlwe.Plaintext, splitAmount)
	encodedVectors2 := make([]*rlwe.Plaintext, splitAmount)

	for i := 0; i < splitAmount; i++ {
		ecd2 := hefloat.NewEncoder(hefloat.Parameters(params))
		var err error
		if err = ecd2.Encode(vectors1[i], encodedVectors1[i]); err != nil {
			panic(err)
		}
		if err = ecd2.Encode(vectors2[i], encodedVectors2[i]); err != nil {
			panic(err)
		}
	}
	encryptedVectors1 := make([]*rlwe.Ciphertext, splitAmount)
	encryptedVectors2 := make([]*rlwe.Ciphertext, splitAmount)
	initials := make([]*rlwe.Ciphertext, splitAmount)
	kgen := hefloat.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	enc := rlwe.NewEncryptor(params, pk)
	for i := 0; i < splitAmount; i++ {
		ciphertext1, err := enc.EncryptNew(encodedVectors1[i])
		if err != nil {
			panic(err)
		}
		encryptedVectors1[i] = ciphertext1
	}
	for i := 0; i < splitAmount; i++ {
		ciphertext2, err := enc.EncryptNew(encodedVectors2[i])
		if err != nil {
			panic(err)
		}
		encryptedVectors2[i] = ciphertext2
	}
	for i := 0; i < splitAmount; i++ {
		evaluator.Mul(encryptedVectors1[i], encryptedVectors2[i], encryptedVectors1[i])
		initials[i] = encryptedVectors1[i].CopyNew()
	}
	for i := 0; i < len(encryptedVectors1); i++ {
		if i != len(encryptedVectors1)-1 {
			for j := 1; j < colnumber1; j++ {
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i])
			}
		} else {
			for j := 1; j < lastnumber1; j++ {
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i])
			}
		}
	}

	for i := 1; i < len(encryptedVectors1); i++ {
		evaluator.Add(encryptedVectors1[0], encryptedVectors1[i], encryptedVectors1[0])
	}

	var btpEvaluator *bootstrapping.Evaluator
	ciphertext3, err := btpEvaluator.Bootstrap(encryptedVectors1[0])
	if err != nil {
		panic(err)
	}
	if err := encoder.Decode(decryptor.DecryptNew(ciphertext3), encryptedVectors1[0]); err != nil {
		panic(err)
	}

	resultFloat := make([]float64, len(encryptedVectors1))

	return resultFloat[:rowNumMatrix]
}

func splitData(data []float64, layerLength, logSlots int) ([][]float64, int, int) {
	columnAmountOnSubbarray := int(math.Floor(float64(logSlots) / float64(layerLength)))
	numSubarrays := int(math.Ceil(float64(len(data)) / (float64(columnAmountOnSubbarray) * float64(layerLength))))

	subarrays := make([][]float64, numSubarrays)

	for i := range subarrays {
		subarrays[i] = make([]float64, logSlots)
	}

	for i := 0; i < numSubarrays; i++ {
		start := i * (columnAmountOnSubbarray * layerLength)
		end := start + (columnAmountOnSubbarray * layerLength)
		if end > len(data) {
			end = len(data)
		}
		subarrays[i] = data[start:end]
	}

	return subarrays, len(subarrays[0]) / layerLength, len(subarrays[len(subarrays)-1]) / layerLength
}

func addRandomNoise(matrix mat.Matrix, noiseAmplitude float64) mat.Matrix {
	rand.NewSource(time.Now().UnixNano())

	r, c := matrix.Dims()
	noisyMatrix := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			noise := (rand.Float64() * 2 * noiseAmplitude) - noiseAmplitude
			value := matrix.At(i, j)
			noisyMatrix.Set(i, j, value+noise)
		}
	}
	return noisyMatrix
}
