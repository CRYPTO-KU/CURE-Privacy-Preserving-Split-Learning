package main

import (
    "fmt"
    "math"

	//"sync"

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

// Sigmoid function
func sigmoid(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}



// Compute the cost for logistic regression
func computeCost(X *mat.Dense, y *mat.VecDense, theta *mat.VecDense) float64 {
    m, _ := X.Dims()
    var cost float64
    epsilon := 1e-15 // Small value to ensure logarithm stability
    for i := 0; i < m; i++ {
        xi := X.RowView(i)
        z := mat.Dot(xi, theta)
        h := sigmoid(z)
        yi := y.AtVec(i)
        cost += -yi*math.Log(h+epsilon) - (1-yi)*math.Log(1-h+epsilon)
    }
    return cost / float64(m)
}

// Compute the gradient for logistic regression (server part)
func computeGradientServerPart(X *mat.Dense, theta *mat.VecDense) *mat.Dense {
    m, _ := X.Dims()
    Z := mat.NewDense(m, 1, nil)
    for i := 0; i < m; i++ {
        xi := X.RowView(i)
        z := mat.Dot(xi, theta)
        Z.Set(i, 0, z)
    }
    return Z
}

func computeGradientServerPartEncrypted(X *mat.Dense, theta *mat.VecDense, cryptoSystem CryptoSystem) *mat.Dense {
    rows, cols := X.Dims()
    Z := mat.NewDense(rows, 1, nil)

    pt2 := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
    if err := cryptoSystem.encoder.Encode(theta.RawVector().Data, pt2); err != nil {
        panic(err)
    }

    ct2, _ := cryptoSystem.encryptor.EncryptNew(pt2)

    //var wg sync.WaitGroup
    //wg.Add(rows)

    for i := 0; i < rows; i++ {
        //go func(i int) {
            //defer wg.Done()
            xi := GetRowAsFloat(X,i)

            pt1 := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
            if err := cryptoSystem.encoder.Encode(xi, pt1); err != nil {
                panic(err)
            }

            ct1, _ := cryptoSystem.encryptor.EncryptNew(pt1)

            sum, _ := inner_product(ct1, ct2, cols, cryptoSystem)
            Z.Set(i, 0, sum)
        //}
    }

    //wg.Wait()
    return Z
}

// Client function to compute the sigmoid
func computeSigmoidClientPart(Z *mat.Dense) *mat.Dense {
    r, _ := Z.Dims()
    H := mat.NewDense(r, 1, nil)
    for i := 0; i < r; i++ {
        z := Z.At(i, 0)
        h := sigmoid(z)
        H.Set(i, 0, h)
    }
    return H
}

// Complete gradient computation
func computeGradient(X *mat.Dense, y *mat.VecDense, theta *mat.VecDense, H *mat.Dense) *mat.VecDense {
    m, n := X.Dims()
    grad := mat.NewVecDense(n, nil)
    for i := 0; i < m; i++ {
        yi := y.AtVec(i)
        hi := H.At(i, 0)
        for j := 0; j < n; j++ {
            grad.SetVec(j, grad.AtVec(j)+(hi-yi)*X.At(i, j))
        }
    }
    for j := 0; j < n; j++ {
        grad.SetVec(j, grad.AtVec(j)/float64(m))
    }
    return grad
}

// Gradient descent optimization
func gradientDescent(X *mat.Dense, y *mat.VecDense, theta *mat.VecDense, alpha float64, numIter int,cryptoSystem CryptoSystem) *mat.VecDense {
    for i := 0; i < numIter; i++ {
        //Z := computeGradientServerPart(X, theta) // Server computes z
		Z := computeGradientServerPartEncrypted(X,theta,cryptoSystem)
        H := computeSigmoidClientPart(Z)        // Client computes sigmoid
        grad := computeGradient(X, y, theta, H) // Server completes gradient computation
        for j := 0; j < theta.Len(); j++ {
            theta.SetVec(j, theta.AtVec(j)-alpha*grad.AtVec(j))
        }
        if i%100 == 0 {
            cost := computeCost(X, y, theta)
            fmt.Printf("Iteration %d, Cost: %f\n", i, cost)
        }
    }
    return theta
}

func inner_product(ct1, ct2 *rlwe.Ciphertext, length_of_ct int, CryptoSystem CryptoSystem) (result float64, err error) {

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
	result_product := mask_first(ct_result, CryptoSystem)
	return result_product, nil
}

func GetRowAsFloat(matrix *mat.Dense, rowIndex int) []float64 {
	_, cols := matrix.Dims()
	row := make([]float64, cols)
	for j := 0; j < cols; j++ {
		row[j] = matrix.At(rowIndex, j)
	}
	return row
}


func mask_first(ct *rlwe.Ciphertext, cryptoSystem CryptoSystem) float64 {
	vect1 := []complex128{0 + 0i}
	pt := cryptoSystem.decyrptor.DecryptNew(ct)
	cryptoSystem.encoder.Decode(pt, vect1)
	return float64(real(vect1[0]))
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

    // Example dataset
    data := []float64{
        1, 34.6, 78.0,
        1, 30.1, 43.7,
        1, 35.3, 72.0,
        1, 60.4, 86.5,
        1, 79.6, 91.2,
    }
    X := mat.NewDense(5, 3, data)
    y := mat.NewVecDense(5, []float64{0, 0, 0, 1, 1})

    // Initialize parameters
    theta := mat.NewVecDense(3, nil)

    // Hyperparameters
    alpha := 0.001 
    numIter := 1000

    // Run gradient descent
    theta = gradientDescent(X, y, theta, alpha, numIter,cryptoSystem)

    // Output the result
    fmt.Printf("Theta: %v\n", theta.RawVector().Data)
}
