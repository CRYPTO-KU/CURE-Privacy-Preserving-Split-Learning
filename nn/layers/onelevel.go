package layers

import (
	"fmt"

	"cure_lib/core/ckkswrapper"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// OneLevelBatchMul computes W·x using packed-batch method.
// - W: plaintext weight matrix [outDim × inDim]
// - x: slice of inDim encrypted vectors (*rlwe.Ciphertext), each packing up to inDim slots
// Returns y: slice of outDim encrypted values, one per output neuron.
func OneLevelBatchMul(W [][]float64, x []*rlwe.Ciphertext, heCtx *ckkswrapper.HeContext) ([]*rlwe.Ciphertext, error) {
	if len(W) == 0 || len(W[0]) == 0 {
		return nil, fmt.Errorf("weight matrix W must be non-empty")
	}
	outDim := len(W)
	inDim := len(W[0])
	if len(x) == 0 {
		return nil, fmt.Errorf("input x must be non-empty")
	}
	eval := ckks.NewEvaluator(heCtx.Params, nil)
	encoder := heCtx.Encoder
	y := make([]*rlwe.Ciphertext, outDim)
	slots := heCtx.Params.MaxSlots()
	batchSize := inDim / len(x)
	if inDim%len(x) != 0 {
		return nil, fmt.Errorf("inDim must be divisible by number of ciphertexts in x")
	}

	// For each output neuron j, compute y_j = sum_i W[j][i] * x[batch][slot] (no slot aggregation)
	for j := 0; j < outDim; j++ {
		var ctSum *rlwe.Ciphertext
		for i := 0; i < inDim; i++ {
			batchIdx := i / batchSize
			slotIdx := i % batchSize
			if batchIdx >= len(x) {
				return nil, fmt.Errorf("batchIdx out of range")
			}
			// Create a mask with 1 at slotIdx, 0 elsewhere
			mask := make([]complex128, slots)
			mask[slotIdx] = complex(W[j][i], 0)
			pt := ckks.NewPlaintext(heCtx.Params, x[batchIdx].Level())
			pt.Scale = x[batchIdx].Scale
			if err := encoder.Encode(mask, pt); err != nil {
				return nil, fmt.Errorf("encode mask failed: %w", err)
			}
			// Multiply x[batchIdx] by mask
			ctProd, err := eval.MulNew(x[batchIdx], pt)
			if err != nil {
				return nil, fmt.Errorf("MulNew failed: %w", err)
			}
			if ctSum == nil {
				ctSum = ctProd
			} else {
				ctSum, err = eval.AddNew(ctSum, ctProd)
				if err != nil {
					return nil, fmt.Errorf("AddNew failed: %w", err)
				}
			}
		}
		y[j] = ctSum
	}
	return y, nil
}

// OneLevelScalarMul computes W·x by repeated scalar multiplies.
// - W: plaintext weight matrix [outDim × inDim]
// - x: single encrypted vector of length inDim (one slot per input coordinate)
// Returns y: slice of outDim encrypted values.
func OneLevelScalarMul(W [][]float64, x *rlwe.Ciphertext, heCtx *ckkswrapper.HeContext) ([]*rlwe.Ciphertext, error) {
	if len(W) == 0 || len(W[0]) == 0 {
		return nil, fmt.Errorf("weight matrix W must be non-empty")
	}
	outDim := len(W)
	inDim := len(W[0])
	eval := ckks.NewEvaluator(heCtx.Params, nil)
	encoder := heCtx.Encoder
	slots := heCtx.Params.MaxSlots()
	y := make([]*rlwe.Ciphertext, outDim)

	for j := 0; j < outDim; j++ {
		var ctSum *rlwe.Ciphertext
		for i := 0; i < inDim; i++ {
			// Create a plaintext vector with W[j][i] at slot i, 0 elsewhere
			mask := make([]complex128, slots)
			mask[i] = complex(W[j][i], 0)
			pt := ckks.NewPlaintext(heCtx.Params, x.Level())
			pt.Scale = x.Scale
			if err := encoder.Encode(mask, pt); err != nil {
				return nil, fmt.Errorf("encode mask failed: %w", err)
			}
			// Multiply x by mask (slot-wise scalar multiply)
			ctProd, err := eval.MulNew(x, pt)
			if err != nil {
				return nil, fmt.Errorf("MulNew failed: %w", err)
			}
			if ctSum == nil {
				ctSum = ctProd
			} else {
				ctSum, err = eval.AddNew(ctSum, ctProd)
				if err != nil {
					return nil, fmt.Errorf("AddNew failed: %w", err)
				}
			}
		}
		y[j] = ctSum
	}
	return y, nil
}

// OneLevelBatchUpdate applies a learning‐rate update to W using encrypted gradients.
// - W: plaintext weight matrix [outDim][inDim]
// - grad: slice of ciphertexts from client, one per output neuron, each packing all inDim slot‐gradients
// - lr: learning rate (float64)
// - heCtx: HE context
func OneLevelBatchUpdate(W [][]float64, grad []*rlwe.Ciphertext, lr float64, heCtx *ckkswrapper.HeContext) error {
	if len(W) == 0 || len(W[0]) == 0 {
		return fmt.Errorf("weight matrix W must be non-empty")
	}
	outDim := len(W)
	inDim := len(W[0])
	if len(grad) != outDim {
		return fmt.Errorf("expected %d gradient ciphertexts, got %d", outDim, len(grad))
	}
	for j := 0; j < outDim; j++ {
		ptGrad := heCtx.Decryptor.DecryptNew(grad[j])
		vecGrad := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(ptGrad, vecGrad)
		for i := 0; i < inDim; i++ {
			W[j][i] -= lr * real(vecGrad[i])
		}
	}
	return nil
}

// OneLevelScalarUpdate applies update for the scalar variant.
// - gradCt: single ciphertext packing all (outDim×inDim) gradients in slots
func OneLevelScalarUpdate(W [][]float64, gradCt *rlwe.Ciphertext, lr float64, heCtx *ckkswrapper.HeContext) error {
	if len(W) == 0 || len(W[0]) == 0 {
		return fmt.Errorf("weight matrix W must be non-empty")
	}
	outDim := len(W)
	inDim := len(W[0])
	ptGrad := heCtx.Decryptor.DecryptNew(gradCt)
	vecGrad := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(ptGrad, vecGrad)
	k := 0
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			W[j][i] -= lr * real(vecGrad[k])
			k++
		}
	}
	return nil
}

// OneLevelLayer wraps the OneLevel functions for benchmarking as a layer

// OneLevelLayer is a struct for benchmarking
// Only implements HE path for forward/update

type OneLevelLayer struct {
	W     [][]float64
	heCtx *ckkswrapper.HeContext
}

func NewOneLevelLayer(W [][]float64, heCtx *ckkswrapper.HeContext) *OneLevelLayer {
	return &OneLevelLayer{W: W, heCtx: heCtx}
}

// ForwardHEIface implements the interface for benchmarking
func (o *OneLevelLayer) ForwardHEIface(x interface{}) (interface{}, error) {
	cts, ok := x.([]*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected []*rlwe.Ciphertext for HE input")
	}
	return OneLevelBatchMul(o.W, cts, o.heCtx)
}

// BackwardHEIface is not implemented (no backward for OneLevel)
func (o *OneLevelLayer) BackwardHEIface(g interface{}) (interface{}, error) {
	return nil, fmt.Errorf("OneLevelLayer does not implement backward HE")
}

// UpdateHE applies encrypted update
func (o *OneLevelLayer) UpdateHE(lr float64, grad interface{}) error {
	cts, ok := grad.([]*rlwe.Ciphertext)
	if !ok {
		return fmt.Errorf("expected []*rlwe.Ciphertext for HE grad")
	}
	return OneLevelBatchUpdate(o.W, cts, lr, o.heCtx)
}
