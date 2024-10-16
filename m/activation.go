package m

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"gonum.org/v1/gonum/mat"
)

type Activator interface {
	Activate(i, j int, sum float64) float64
	Deactivate(m mat.Matrix) mat.Matrix
	fmt.Stringer
}

var ActivatorLookup = map[string]Activator{
	"sigmoid": Sigmoid{},
	"tanh":    Tanh{},
	"relu":    ReLU{},
}

type Sigmoid struct{}

func (s Sigmoid) Activate(i, j int, sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

func (s Sigmoid) Deactivate(matrix mat.Matrix) mat.Matrix {
	rows, _ := matrix.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(matrix, subtract(ones, matrix))
}

func (s Sigmoid) String() string {
	return "sigmoid"
}

type Tanh struct{}

func (t Tanh) Activate(i, j int, sum float64) float64 {
	return math.Tanh(sum)
}

func (t Tanh) Deactivate(matrix mat.Matrix) mat.Matrix {
	tanhPrime := func(i, j int, v float64) float64 {
		return 1.0 - (math.Tanh(v) * math.Tanh(v))
	}

	return apply(tanhPrime, matrix)
}

func (t Tanh) String() string {
	return "tanh"
}

type ReLU struct{} // Define the ReLU activation type.

func (r ReLU) Activate(i, j int, sum float64) float64 {
	if sum < 0 {
		return 0.0001 * sum
	}
	return sum
}

func (r ReLU) Deactivate(matrix mat.Matrix) mat.Matrix {
	applyReLU := func(i, j int, v float64) float64 {
		if v < 0 {
			return 0.0001
		}
		return 1
	}
	return apply(applyReLU, matrix)
}

func (r ReLU) String() string {
	return "relu"
}

func ckksActivateSigmoid(m *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext { //0.4989,0.2146,−0.0373 for first 3 Chebyshev polys
	x_1 := ckksScale(m, 0.197, evaluator)
	x_3 := ckksMultiply(m, m, evaluator)
	x_3 = ckksMultiply(x_3, m, evaluator)
	x_3 = ckksScale(x_3, -0.004, evaluator)
	x_1 = ckksAddScalar(x_1, 0.5, evaluator)
	x_1 = ckksAdd(x_1, x_3, evaluator)

	return x_1
}

func ckksDeactivateSigmoid(m *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext { // using the analytical result of the sigmoid function (ie s' = s(1-s))
	s_x := ckksActivateSigmoid(m, evaluator)
	minus_s_x := ckksScale(s_x, -1, evaluator)
	minus_s_x = ckksAddScalar(minus_s_x, 1, evaluator)
	s_x = ckksMultiply(s_x, minus_s_x, evaluator)

	return s_x
}
