//go:build debug
// +build debug

package layers

import (
	"cure_lib/tensor"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func (a *Activation) DebugCompareForward(result interface{}, t *testing.T) {
	if t != nil {
		outputShadow := tensor.New(len(a.lastInputShadow.Data))
		coeffs := a.poly.Coeffs
		degree := a.poly.Degree

		for i, val := range a.lastInputShadow.Data {
			res := coeffs[degree]
			for j := degree - 1; j >= 0; j-- {
				res = res*val + coeffs[j]
			}
			outputShadow.Data[i] = res
		}

		if ct, ok := result.(*rlwe.Ciphertext); ok {
			a.heCtx.DebugCompare(ct, outputShadow, "After Activation Forward", 1e-4, t)
		}
	}
}
