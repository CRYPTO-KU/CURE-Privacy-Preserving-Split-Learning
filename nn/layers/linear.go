package layers

import (
	"fmt"

	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Linear is a fully-connected layer supporting plaintext & HE-masked forward.
type Linear struct {
	// plaintext params
	W, B *tensor.Tensor

	// HE params
	heCtx     *ckkswrapper.HeContext
	serverKit *ckkswrapper.ServerKit

	// pre-encrypted CKKS weights, plus bias/plain mask
	weightCTs   []*rlwe.Ciphertext // row-wise encrypted weights
	biasPT      *rlwe.Plaintext    // plaintext bias (kept at default scale)
	maskPT      *rlwe.Plaintext    // one-hot 〈1,0,0,…〉 for slot-0 extraction
	weightT_PTs []*rlwe.Plaintext  // Plaintext weights, transposed (column-wise)

	encrypted bool

	lastInput   *rlwe.Ciphertext
	weightGrads []*rlwe.Ciphertext

	// -- Shadow plaintext for debugging (used in debug builds only) --
	wShadow         *tensor.Tensor
	bShadow         *tensor.Tensor
	lastInputShadow *tensor.Tensor

	weightGradsPlain []*tensor.Tensor        // for plaintext gradients
	lrPlainCache     map[int]*rlwe.Plaintext // keyed by level
}

// NewLinear(inDim→outDim, encrypted, heCtx) sets up W,B and HE context.
func NewLinear(inDim, outDim int, encrypted bool, heCtx *ckkswrapper.HeContext) *Linear {
	l := &Linear{W: tensor.New(outDim, inDim), B: tensor.New(outDim), encrypted: encrypted, heCtx: heCtx}

	// Initialize shadow plaintext fields
	l.wShadow = tensor.New(outDim, inDim)
	copy(l.wShadow.Data, l.W.Data)
	l.bShadow = tensor.New(outDim)
	copy(l.bShadow.Data, l.B.Data)

	if encrypted {
		// collect only the needed rotations:
		//  - powers-of-two for treeSum in forward pass (input dimension)
		//  - one rotation by j for each output slot j (forward pass)
		//  - powers-of-two for treeSum in backward pass (output dimension)
		//  - powers-of-two NEGATIVE for broadcast in backward pass (input dimension)
		rots := []int{}
		// Keys for forward pass dot-product
		for step := 1; step < inDim; step *= 2 {
			rots = append(rots, step)
		}
		// Keys for forward pass assembly
		for j := 0; j < outDim; j++ {
			rots = append(rots, -j)
		}
		// Keys for backward pass dL/dx
		for step := 1; step < outDim; step *= 2 {
			rots = append(rots, step)
		}
		// --- NEW: Keys for backward pass dL/dW broadcastScalar ---
		for step := 1; step < inDim; step *= 2 {
			rots = append(rots, -step)
		}
		// Keys for backward pass dL/dW isolateSlot
		for j := 0; j < outDim; j++ {
			rots = append(rots, j)
		}
		l.serverKit = heCtx.GenServerKit(rots)
		l.weightCTs = make([]*rlwe.Ciphertext, outDim)
		l.weightGrads = make([]*rlwe.Ciphertext, outDim)
		// build once: plaintext 〈1,0,0,…〉 for masking slot-0
		mvec := make([]complex128, heCtx.Params.MaxSlots())
		mvec[0] = 1
		mp := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		l.serverKit.Encoder.Encode(mvec, mp)
		l.maskPT = mp
	}
	return l
}

// EnableEncrypted switches the layer between encrypted and plaintext mode
func (l *Linear) EnableEncrypted(encrypted bool) {
	l.encrypted = encrypted
	if encrypted && l.heCtx != nil && l.serverKit == nil {
		// Initialize HE components if switching to encrypted mode
		inDim, outDim := l.W.Shape[1], l.W.Shape[0]
		rots := []int{}
		// Keys for forward pass dot-product
		for step := 1; step < inDim; step *= 2 {
			rots = append(rots, step)
		}
		// Keys for forward pass assembly
		for j := 0; j < outDim; j++ {
			rots = append(rots, -j)
		}
		// Keys for backward pass dL/dx
		for step := 1; step < outDim; step *= 2 {
			rots = append(rots, step)
		}
		// Keys for backward pass dL/dW broadcastScalar
		for step := 1; step < inDim; step *= 2 {
			rots = append(rots, -step)
		}
		// Keys for backward pass dL/dW isolateSlot
		for j := 0; j < outDim; j++ {
			rots = append(rots, j)
		}
		l.serverKit = l.heCtx.GenServerKit(rots)
		l.weightCTs = make([]*rlwe.Ciphertext, outDim)
		l.weightGrads = make([]*rlwe.Ciphertext, outDim)
		// build once: plaintext 〈1,0,0,…〉 for masking slot-0
		mvec := make([]complex128, l.heCtx.Params.MaxSlots())
		mvec[0] = 1
		mp := ckks.NewPlaintext(l.heCtx.Params, l.heCtx.Params.MaxLevel())
		l.serverKit.Encoder.Encode(mvec, mp)
		l.maskPT = mp
	}
}

// SyncHE encodes W rows, bias, and the slot-0 mask into ciphertexts.
func (l *Linear) SyncHE() {
	if !l.encrypted {
		return
	}
	inDim, outDim := l.W.Shape[1], l.W.Shape[0]
	slots := l.heCtx.Params.MaxSlots()

	// Update shadow weights to match current weights
	copy(l.wShadow.Data, l.W.Data)
	copy(l.bShadow.Data, l.B.Data)

	// encode and encrypt each row of W
	l.weightCTs = make([]*rlwe.Ciphertext, outDim)
	for j := 0; j < outDim; j++ {
		wrow := make([]complex128, slots)
		for i := 0; i < inDim; i++ {
			wrow[i] = complex(l.W.Data[j*inDim+i], 0)
		}
		pt := ckks.NewPlaintext(l.heCtx.Params, l.heCtx.Params.MaxLevel())
		pt.Scale = l.heCtx.Params.DefaultScale()
		l.serverKit.Encoder.Encode(wrow, pt)
		ct, err := l.heCtx.Encryptor.EncryptNew(pt)
		if err != nil {
			panic(err)
		}
		l.weightCTs[j] = ct
	}

	// encode bias vector – keep as *plaintext* so we can add after final rescale
	bvec := make([]complex128, slots)
	for j := 0; j < outDim; j++ {
		bvec[j] = complex(l.B.Data[j], 0)
	}
	bp := ckks.NewPlaintext(l.heCtx.Params, l.heCtx.Params.MaxLevel())
	l.serverKit.Encoder.Encode(bvec, bp)
	l.biasPT = bp

	// NEW: Encode and store transposed weights (columns) as plaintexts
	l.weightT_PTs = make([]*rlwe.Plaintext, inDim)
	for i := 0; i < inDim; i++ { // Iterate through columns of W
		w_col := make([]complex128, slots)
		for j := 0; j < outDim; j++ { // Iterate through rows of W
			w_col[j] = complex(l.W.Data[j*inDim+i], 0)
		}
		pt := ckks.NewPlaintext(l.heCtx.Params, l.heCtx.Params.MaxLevel())
		l.serverKit.Encoder.Encode(w_col, pt)
		l.weightT_PTs[i] = pt
	}
}

// treeSumCT multiplies ctX by ctW, then tree-sums into slot 0.
func (l *Linear) treeSumCT(ctX, ctW *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	// 1) multiply ciphertexts (degree 2)
	tmp, err := l.serverKit.Evaluator.MulNew(ctX, ctW)
	if err != nil {
		return nil, err
	}
	// 2) relinearize back to degree 1
	tmp, _ = l.serverKit.Evaluator.RelinearizeNew(tmp)

	// 3) rescale → drop one modulus, restore default scale
	out := rlwe.NewCiphertext(l.serverKit.Params, tmp.Degree(), tmp.Level()-1)
	if err := l.serverKit.Evaluator.Rescale(tmp, out); err != nil {
		return nil, err
	}
	tmp = out

	// 4) tree-sum rotations
	inDim := l.W.Shape[1]
	for step := 1; step < inDim; step *= 2 {
		rot, err := l.serverKit.Evaluator.RotateNew(tmp, step)
		if err != nil {
			return nil, err
		}
		tmp, err = l.serverKit.Evaluator.AddNew(tmp, rot)
		if err != nil {
			return nil, err
		}
	}
	return tmp, nil
}

// ForwardCipherMasked returns y = W·x + B as one ciphertext:
func (l *Linear) ForwardCipherMasked(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if !l.encrypted {
		return nil, fmt.Errorf("ForwardCipherMasked on plaintext layer")
	}
	l.lastInput = ct

	outDim := l.W.Shape[0]
	workerEval := l.serverKit.GetWorkerEvaluator()

	// Assemble output using plaintext packing
	outputVec := make([]complex128, l.heCtx.Params.MaxSlots())
	for j := 0; j < outDim; j++ {
		// (1) ct·ct dot → slot-0 everywhere
		dot, err := l.treeSumCTWithEval(ct, l.weightCTs[j], workerEval)
		if err != nil {
			return nil, err
		}

		// (2) rotate slot-0 into slot j
		rot, err := workerEval.RotateNew(dot, -j)
		if err != nil {
			return nil, err
		}

		// (3) mask with one-hot at slot j
		oneHot := make([]complex128, l.heCtx.Params.MaxSlots())
		oneHot[j] = 1
		maskPT := ckks.NewPlaintext(l.heCtx.Params, rot.Level())
		maskPT.Scale = rot.Scale
		l.heCtx.Encoder.Encode(oneHot, maskPT)
		masked, err := workerEval.MulNew(rot, maskPT)
		if err != nil {
			return nil, err
		}

		// (4) decrypt masked, extract slot j, store in outputVec
		ptMasked := l.heCtx.Decryptor.DecryptNew(masked)
		valsMasked := make([]complex128, l.heCtx.Params.MaxSlots())
		l.heCtx.Encoder.Decode(ptMasked, valsMasked)
		outputVec[j] = valsMasked[j]
	}

	// Encode and encrypt output vector
	ptOut := ckks.NewPlaintext(l.heCtx.Params, l.heCtx.Params.MaxLevel())
	ptOut.Scale = l.heCtx.Params.DefaultScale()
	l.heCtx.Encoder.Encode(outputVec, ptOut)
	resCt, err := l.heCtx.Encryptor.EncryptNew(ptOut)
	if err != nil {
		return nil, err
	}

	// Add bias
	biasVec := make([]complex128, l.heCtx.Params.MaxSlots())
	for j := 0; j < outDim; j++ {
		biasVec[j] = complex(l.B.Data[j], 0)
	}
	biasPT := ckks.NewPlaintext(l.heCtx.Params, resCt.Level())
	biasPT.Scale = resCt.Scale
	l.heCtx.Encoder.Encode(biasVec, biasPT)
	resCt, err = l.serverKit.Evaluator.AddNew(resCt, biasPT)
	if err != nil {
		return nil, err
	}

	l.lastInput = ct.CopyNew()
	return resCt, nil
}

// ForwardPlaintext computes y = Wx + B for a plaintext tensor.
func (l *Linear) ForwardPlaintext(x *tensor.Tensor) (*tensor.Tensor, error) {
	if l.encrypted {
		return nil, fmt.Errorf("ForwardPlaintext on encrypted layer")
	}
	// Cache input for backward
	l.lastInputShadow = tensor.New(len(x.Data))
	copy(l.lastInputShadow.Data, x.Data)
	if len(x.Shape) == 1 {
		// x is a vector, treat as (inDim, 1)
		x = &tensor.Tensor{Data: x.Data, Shape: []int{x.Shape[0], 1}}
	}
	if len(l.W.Shape) != 2 || len(x.Shape) != 2 {
		return nil, fmt.Errorf("expected 2D weights and 2D input, got %v and %v", l.W.Shape, x.Shape)
	}
	wx, err := tensor.MatMul(l.W, x)
	if err != nil {
		return nil, err
	}
	// wx is (outDim, batchSize), l.B is (outDim)
	b := l.B
	if len(b.Shape) == 1 && b.Shape[0] == wx.Shape[0] {
		if len(wx.Shape) == 2 {
			// Broadcast bias across batch
			for i := 0; i < wx.Shape[0]; i++ {
				for j := 0; j < wx.Shape[1]; j++ {
					wx.Data[i*wx.Shape[1]+j] += b.Data[i]
				}
			}
			return wx, nil
		} else if len(wx.Shape) == 2 && wx.Shape[1] == 1 {
			// Add bias to each row (single column)
			for i := 0; i < wx.Shape[0]; i++ {
				wx.Data[i] += b.Data[i]
			}
			return wx, nil
		}
	}
	// Otherwise, try tensor.Add
	return tensor.Add(wx, b)
}

// Forward processes the input through the layer.
// It accepts either a *tensor.Tensor or an *rlwe.Ciphertext.
func (l *Linear) Forward(input interface{}) (interface{}, error) {
	if l.encrypted {
		ctInput, ok := input.(*rlwe.Ciphertext)
		if !ok {
			return nil, fmt.Errorf("encrypted layer expects *rlwe.Ciphertext input")
		}
		return l.ForwardCipherMasked(ctInput)
	}
	ptInput, ok := input.(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("plaintext layer expects *tensor.Tensor input")
	}
	return l.ForwardPlaintext(ptInput)
}

// ForwardHE performs HE forward pass.
func (l *Linear) ForwardHE(x interface{}) (interface{}, error) {
	return l.Forward(x)
}

// BackwardHE performs HE backward pass.
func (l *Linear) BackwardHE(g interface{}) (interface{}, error) {
	return l.backwardHE(g)
}

// BackwardHEIface is a wrapper for BackwardHE to match the interface.
func (l *Linear) BackwardHEIface(g interface{}) (interface{}, error) {
	return l.BackwardHE(g)
}

// UpdateHE applies an in-place SGD step on the encrypted weights:
//
//	weightCTs[j] <- weightCTs[j] - lr * weightGrads[j]
//
// Gradients are assumed to have been populated by BackwardWithDebug/BackwardHE.
// Only the first inDim slots of each row are used; remaining slots pass through.
func (l *Linear) UpdateHE(lr float64) error {
	if !l.encrypted {
		return fmt.Errorf("UpdateHE called on plaintext layer")
	}
	if len(l.weightGrads) == 0 {
		// nothing to do
		return nil
	}

	eval := l.serverKit.Evaluator
	params := l.heCtx.Params
	slots := params.MaxSlots()
	_ = slots // currently unused after encoding

	// Encode (-lr) replicated across all slots at each gradient level as needed.
	// We cache per-level to avoid repeated encodes.
	negLR := -lr

	for j, g := range l.weightGrads {
		if g == nil {
			continue
		}

		// (1) scale gradient by -lr
		lrPT := l.getConstPlain(negLR, g.Level(), g.Scale)
		step, err := eval.MulNew(g, lrPT)
		if err != nil {
			return fmt.Errorf("UpdateHE: MulNew grad*lr (j=%d): %w", j, err)
		}
		if step.Degree() > 1 {
			if step, err = eval.RelinearizeNew(step); err != nil {
				return fmt.Errorf("UpdateHE: Relinearize (j=%d): %w", j, err)
			}
		}
		if err := eval.Rescale(step, step); err != nil {
			return fmt.Errorf("UpdateHE: Rescale grad step (j=%d): %w", j, err)
		}

		// (2) level-align weight row with grad step
		wt := l.weightCTs[j]
		for wt.Level() > step.Level() {
			// drop excess primes from the weight row until levels match
			if err := eval.Rescale(wt, wt); err != nil {
				return fmt.Errorf("UpdateHE: Rescale weight row (j=%d): %w", j, err)
			}
		}
		// (3) unify scale (lightweight assignment; acceptable when magnitudes are close)
		wt.Scale = step.Scale

		// (4) apply update
		upd, err := eval.AddNew(wt, step)
		if err != nil {
			return fmt.Errorf("UpdateHE: Add weight+step (j=%d): %w", j, err)
		}
		l.weightCTs[j] = upd
	}

	// Bias update: if you produce an encrypted bias gradient, apply same pattern here.
	// Currently omitted (biasPT remains as synced).

	return nil
}

// Levels and Encrypted metadata
func (l *Linear) Levels() int {
	if l.encrypted {
		return 1
	}
	return 0
}
func (l *Linear) Encrypted() bool { return l.encrypted }

// Add a public getter for heCtx
func (l *Linear) HeContext() *ckkswrapper.HeContext {
	return l.heCtx
}

// Add a helper for treeSumCT with a custom Evaluator
func (l *Linear) treeSumCTWithEval(ctX, ctW *rlwe.Ciphertext, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	// 1) multiply ciphertexts (degree 2)
	tmp, err := eval.MulNew(ctX, ctW)
	if err != nil {
		return nil, err
	}
	// 2) relinearize back to degree 1
	tmp, _ = eval.RelinearizeNew(tmp)

	// 3) rescale → drop one modulus, restore default scale
	out := rlwe.NewCiphertext(l.serverKit.Params, tmp.Degree(), tmp.Level()-1)
	if err := eval.Rescale(tmp, out); err != nil {
		return nil, err
	}
	tmp = out

	// 4) tree-sum rotations
	inDim := l.W.Shape[1]
	for step := 1; step < inDim; step *= 2 {
		rot, err := eval.RotateNew(tmp, step)
		if err != nil {
			return nil, err
		}
		tmp, err = eval.AddNew(tmp, rot)
		if err != nil {
			return nil, err
		}
	}
	return tmp, nil
}

// backwardHE computes gradients for the HE (encrypted) case.
func (l *Linear) backwardHE(gradOut interface{}) (interface{}, error) {
	if !l.encrypted {
		return nil, fmt.Errorf("backwardHE called on plaintext layer")
	}
	gradOutCt, ok := gradOut.(*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected *rlwe.Ciphertext for gradOut")
	}
	if l.lastInput == nil {
		return nil, fmt.Errorf("no cached input for backward pass")
	}

	inDim, outDim := l.W.Shape[1], l.W.Shape[0]
	workerEval := l.serverKit.GetWorkerEvaluator()

	// 1. Calculate Weight Gradients dL/dW = (dL/dy) * x^T
	l.weightGrads = make([]*rlwe.Ciphertext, outDim)
	for j := 0; j < outDim; j++ {
		gradScalarCt, err := l.IsolateSlot(gradOutCt, j, workerEval)
		if err != nil {
			return nil, fmt.Errorf("failed to isolate slot %d: %w", j, err)
		}

		broadcastCt, err := l.BroadcastScalar(gradScalarCt, inDim, workerEval)
		if err != nil {
			return nil, fmt.Errorf("failed to broadcast slot %d: %w", j, err)
		}

		gradW_j, err := workerEval.MulNew(l.lastInput, broadcastCt)
		if err != nil {
			return nil, fmt.Errorf("failed to multiply input and broadcast for j=%d: %w", j, err)
		}

		if gradW_j, err = workerEval.RelinearizeNew(gradW_j); err != nil {
			return nil, err
		}
		if gradW_j.Level() > 0 {
			if err = workerEval.Rescale(gradW_j, gradW_j); err != nil {
				return nil, err
			}
		}

		l.weightGrads[j] = gradW_j
	}

	// 2. Calculate Input Gradients dL/dx = W^T * (dL/dy)
	gradInPartials := make([]*rlwe.Ciphertext, inDim)
	for k := 0; k < inDim; k++ {
		dot, err := workerEval.MulNew(gradOutCt, l.weightT_PTs[k])
		if err != nil {
			return nil, fmt.Errorf("failed to multiply gradOut and weightT for k=%d: %w", k, err)
		}

		if dot.Level() > 0 {
			if err = workerEval.Rescale(dot, dot); err != nil {
				return nil, err
			}
		}

		// Tree sum over outDim
		result := dot
		for step := 1; step < outDim; step *= 2 {
			rot, err := workerEval.RotateNew(result, step)
			if err != nil {
				return nil, fmt.Errorf("failed to rotate in dL/dx tree sum k=%d, step=%d: %w", k, step, err)
			}
			if err := workerEval.Add(result, rot, result); err != nil {
				return nil, fmt.Errorf("failed to add in dL/dx tree sum k=%d, step=%d: %w", k, step, err)
			}
		}
		gradInPartials[k] = result
	}

	// 3. Assemble the dL/dx ciphertext
	gradInCt := gradInPartials[0]
	for k := 1; k < inDim; k++ {
		if err := workerEval.Add(gradInCt, gradInPartials[k], gradInCt); err != nil {
			return nil, fmt.Errorf("failed to sum gradInPartials k=%d: %w", k, err)
		}
	}

	if gradInCt.Level() > 0 {
		if err := workerEval.Rescale(gradInCt, gradInCt); err != nil {
			return nil, err
		}
	}

	return gradInCt, nil
}

// Backward computes gradients for the plaintext (unencrypted) case.
func (l *Linear) Backward(gradOut interface{}) (interface{}, error) {
	if l.encrypted {
		return nil, fmt.Errorf("Plaintext Backward called on encrypted layer")
	}
	gradOutTensor, ok := gradOut.(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("Expected *tensor.Tensor for gradOut")
	}
	inDim, outDim := l.W.Shape[1], l.W.Shape[0]
	input := l.lastInputShadow
	if input == nil {
		return nil, fmt.Errorf("No cached input for backward pass")
	}

	// Support batched (2D) gradients
	if len(input.Shape) == 2 && len(gradOutTensor.Shape) == 2 && input.Shape[1] == gradOutTensor.Shape[1] {
		batchSize := input.Shape[1]
		gradW := tensor.New(outDim, inDim)
		gradB := tensor.New(outDim)
		gradIn := tensor.New(inDim, batchSize)
		for b := 0; b < batchSize; b++ {
			for j := 0; j < outDim; j++ {
				idx := j*batchSize + b
				if idx >= len(gradOutTensor.Data) {
					return nil, fmt.Errorf("Index out of bounds: j=%d, batchSize=%d, b=%d, idx=%d, gradOutTensor.Data len=%d", j, batchSize, b, idx, len(gradOutTensor.Data))
				}
				g := gradOutTensor.Data[idx]
				gradB.Data[j] += g
				for i := 0; i < inDim; i++ {
					gradW.Data[j*inDim+i] += g * input.Data[i*batchSize+b]
				}
			}
		}
		// Average gradients over batch
		for i := range gradW.Data {
			gradW.Data[i] /= float64(batchSize)
		}
		for i := range gradB.Data {
			gradB.Data[i] /= float64(batchSize)
		}
		l.weightGradsPlain = []*tensor.Tensor{gradW, gradB}
		// dL/dx = W^T * gradOut, for each sample
		for b := 0; b < batchSize; b++ {
			for i := 0; i < inDim; i++ {
				val := 0.0
				for j := 0; j < outDim; j++ {
					idx := j*batchSize + b
					if idx >= len(gradOutTensor.Data) {
						return nil, fmt.Errorf("Index out of bounds in gradIn computation: j=%d, batchSize=%d, b=%d, idx=%d, gradOutTensor.Data len=%d", j, batchSize, b, idx, len(gradOutTensor.Data))
					}
					val += l.W.Data[j*inDim+i] * gradOutTensor.Data[idx]
				}
				gradIn.Data[i*batchSize+b] = val
			}
		}
		return gradIn, nil
	}
	// If input is [inDim, batchSize] but gradOut is [outDim], broadcast gradOut to [outDim, batchSize]
	if len(input.Shape) == 2 && len(gradOutTensor.Shape) == 1 && gradOutTensor.Shape[0] == outDim {
		batchSize := input.Shape[1]
		gradOutMat := tensor.New(outDim, batchSize)
		for j := 0; j < outDim; j++ {
			for b := 0; b < batchSize; b++ {
				gradOutMat.Data[j*batchSize+b] = gradOutTensor.Data[j]
			}
		}
		return l.Backward(gradOutMat)
	}
	// If input is [inDim, batchSize] but gradOut is [outDim, 1], expand to [outDim, batchSize]
	if len(input.Shape) == 2 && len(gradOutTensor.Shape) == 2 && gradOutTensor.Shape[0] == outDim && gradOutTensor.Shape[1] == 1 {
		batchSize := input.Shape[1]
		gradOutMat := tensor.New(outDim, batchSize)
		for j := 0; j < outDim; j++ {
			for b := 0; b < batchSize; b++ {
				gradOutMat.Data[j*batchSize+b] = gradOutTensor.Data[j]
			}
		}
		return l.Backward(gradOutMat)
	}
	// Fallback: original (flat) case
	// dL/dW = gradOut * input^T (outer product)
	gradW := tensor.New(outDim, inDim)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			gradW.Data[j*inDim+i] = gradOutTensor.Data[j] * input.Data[i]
		}
	}
	l.weightGradsPlain = []*tensor.Tensor{gradW}
	// dL/dx = W^T * gradOut
	gradIn := tensor.New(inDim)
	for i := 0; i < inDim; i++ {
		val := 0.0
		for j := 0; j < outDim; j++ {
			val += l.W.Data[j*inDim+i] * gradOutTensor.Data[j]
		}
		gradIn.Data[i] = val
	}
	return gradIn, nil
}

// IsolateSlot takes a ciphertext `ct` and an index `j`, and returns a new
// ciphertext where slot 0 contains the value from ct's slot j, and all other slots are zero.
func (l *Linear) IsolateSlot(ct *rlwe.Ciphertext, j int, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	// 1. Rotate to bring the target slot to position 0 (use negative rotation)
	rotatedCt, err := eval.RotateNew(ct, -j)
	if err != nil {
		return nil, fmt.Errorf("isolateSlot negative rotate failed: %w", err)
	}

	// 2. Ensure rotatedCt is at default scale
	defaultScale := l.heCtx.Params.DefaultScale().Float64()
	if rotatedCt.Scale.Float64() != defaultScale {
		if rotatedCt.Level() > 0 {
			if err = eval.Rescale(rotatedCt, rotatedCt); err != nil {
				return nil, err
			}
		} else {
			rotatedCt = l.heCtx.Refresh(rotatedCt)
		}
		rotatedCt.Scale = l.heCtx.Params.DefaultScale()
	}

	// 3. Create the mask at default scale using the layer's encoder
	maskVec := make([]complex128, l.heCtx.Params.MaxSlots())
	maskVec[0] = 1
	maskPT := ckks.NewPlaintext(l.heCtx.Params, rotatedCt.Level())
	maskPT.Scale = l.heCtx.Params.DefaultScale()
	if err := l.serverKit.Encoder.Encode(maskVec, maskPT); err != nil {
		return nil, fmt.Errorf("isolateSlot mask encoding failed: %w", err)
	}

	// 4. Multiply to perform the masking.
	isolatedCt, err := eval.MulNew(rotatedCt, maskPT)
	if err != nil {
		return nil, fmt.Errorf("isolateSlot multiplication failed: %w", err)
	}

	// 5. Relinearize if needed
	if isolatedCt.Degree() > 1 {
		if err = eval.Relinearize(isolatedCt, isolatedCt); err != nil {
			return nil, err
		}
	}

	// 6. Rescale to default scale if possible, else refresh
	if isolatedCt.Scale.Float64() != defaultScale {
		if isolatedCt.Level() > 0 {
			if err = eval.Rescale(isolatedCt, isolatedCt); err != nil {
				return nil, err
			}
		} else {
			isolatedCt = l.heCtx.Refresh(isolatedCt)
		}
		isolatedCt.Scale = l.heCtx.Params.DefaultScale()
	}

	return isolatedCt, nil
}

// BroadcastScalar takes a ciphertext with a value in slot 0 and broadcasts it to the first n slots using negative rotations.
func (l *Linear) BroadcastScalar(ct *rlwe.Ciphertext, n int, eval *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	broadcastCt := ct.CopyNew()
	for i := 1; i < n; i <<= 1 {
		rotated, err := eval.RotateNew(broadcastCt, -i)
		if err != nil {
			return nil, fmt.Errorf("broadcastScalar negative rotate failed: %w", err)
		}
		if err = eval.Add(broadcastCt, rotated, broadcastCt); err != nil {
			return nil, fmt.Errorf("broadcastScalar add failed: %w", err)
		}
	}
	return broadcastCt, nil
}

// Update applies the calculated gradients to the weights (plaintext only).
func (l *Linear) Update(learningRate float64) error {
	if l.encrypted {
		return nil // No-op for encrypted, handled by UpdateHE
	}
	if len(l.weightGradsPlain) == 0 {
		return fmt.Errorf("no gradients to update")
	}
	gradW := l.weightGradsPlain[0]
	inDim, outDim := l.W.Shape[1], l.W.Shape[0]
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			l.W.Data[j*inDim+i] -= learningRate * gradW.Data[j*inDim+i]
		}
	}
	// Update bias if present
	if len(l.weightGradsPlain) > 1 {
		gradB := l.weightGradsPlain[1]
		for j := 0; j < outDim; j++ {
			l.B.Data[j] -= learningRate * gradB.Data[j]
		}
	}
	return nil
}

// GetEvaluator returns the server kit evaluator for testing purposes.
func (l *Linear) GetEvaluator() *ckks.Evaluator {
	return l.serverKit.Evaluator
}

// Helper: encode replicated scalar at a given level/scale
func (l *Linear) getConstPlain(val float64, level int, scale rlwe.Scale) *rlwe.Plaintext {
	if l.lrPlainCache == nil {
		l.lrPlainCache = make(map[int]*rlwe.Plaintext)
	}
	if pt, ok := l.lrPlainCache[level]; ok {
		// NOTE: assumes cached value already encoded with `val`; recreate if changing sign/val
		return pt
	}
	slots := l.heCtx.Params.MaxSlots()
	vec := make([]complex128, slots)
	for i := range vec {
		vec[i] = complex(val, 0)
	}
	pt := ckks.NewPlaintext(l.heCtx.Params, level)
	pt.Scale = scale
	l.serverKit.Encoder.Encode(vec, pt)
	l.lrPlainCache[level] = pt
	return pt
}

func (l *Linear) Tag() string {
	inDim := l.W.Shape[1]
	outDim := l.W.Shape[0]
	return fmt.Sprintf("Linear_%d_%d", inDim, outDim)
}
