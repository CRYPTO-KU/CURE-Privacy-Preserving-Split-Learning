package layers

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Poly holds the definition of a polynomial approximation.
type Poly struct {
	Name   string
	Coeffs []float64
	Degree int
	Levels int // Levels consumed by the HE evaluation
}

// Activation is a layer that applies a polynomial function.
type Activation struct {
	poly      Poly
	encrypted bool
	heCtx     *ckkswrapper.HeContext
	serverKit *ckkswrapper.ServerKit
	lastInput *rlwe.Ciphertext

	// -- Shadow plaintext for debugging --
	lastInputShadow *tensor.Tensor
}

// SupportedPolynomials contains precomputed polynomial approximations.
var SupportedPolynomials = map[string]Poly{
	"ReLU3": {
		Name:   "ReLU3",
		Coeffs: []float64{0.3183099, 0.5, 0.2122066, 0},
		Degree: 3,
		Levels: 2,
	},
	"ReLU3_deriv": {
		Name:   "ReLU3_deriv",
		Coeffs: []float64{0.5, 0.4244}, // c0, c1
		Degree: 1,
		Levels: 1,
	},
}

// NewActivation creates a new activation layer.
func NewActivation(polyName string, encrypted bool, heCtx *ckkswrapper.HeContext) (*Activation, error) {
	poly, ok := SupportedPolynomials[polyName]
	if !ok {
		return nil, fmt.Errorf("unsupported polynomial: %s", polyName)
	}

	a := &Activation{
		poly:      poly,
		encrypted: encrypted,
		heCtx:     heCtx,
	}

	if encrypted {
		if heCtx == nil {
			return nil, fmt.Errorf("heCtx is required for an encrypted activation layer")
		}
		// Always re-initialize serverKit, even if already set
		a.serverKit = heCtx.GenServerKit([]int{})
	}
	return a, nil
}

func (a *Activation) Levels() int {
	if a.encrypted {
		return a.poly.Levels
	}
	return 0
}

func (a *Activation) Encrypted() bool {
	return a.encrypted
}

func (a *Activation) EnableEncrypted(encrypted bool) {
	a.encrypted = encrypted
	if encrypted && a.heCtx != nil {
		a.serverKit = a.heCtx.GenServerKit([]int{})
	}
}

// Backward computes gradients for the plaintext (unencrypted) case.
func (a *Activation) Backward(gradOut interface{}) (interface{}, error) {
	if a.encrypted {
		return nil, fmt.Errorf("Plaintext Backward called on encrypted activation layer")
	}
	gradOutTensor, ok := gradOut.(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("Expected *tensor.Tensor for gradOut")
	}
	input := a.lastInputShadow
	if input == nil {
		return nil, fmt.Errorf("No cached input for backward pass")
	}
	if a.poly.Name == "ReLU3" {
		gradIn := tensor.New(len(input.Data))
		for i := range gradIn.Data {
			deriv := 0.0
			if input.Data[i] > 0 {
				deriv = 1.0
			}
			gradIn.Data[i] = gradOutTensor.Data[i] * deriv
		}
		return gradIn, nil
	}
	// Support batched (2D) input, robust to shape mismatches
	if len(input.Data) == len(gradOutTensor.Data) {
		// Flat, just elementwise
		gradIn := tensor.New(len(input.Data))
		c0 := 0.5
		c1 := 0.4244
		for i := range gradIn.Data {
			deriv := c0 + c1*input.Data[i]
			gradIn.Data[i] = gradOutTensor.Data[i] * deriv
		}
		return gradIn, nil
	}
	// Handle case where input is flattened [inDim*batchSize] but gradOut is [outDim]
	// This happens when Linear layer returns [outDim] instead of [outDim, batchSize]
	if len(input.Shape) == 1 && len(gradOutTensor.Shape) == 1 {
		// Try to infer batch size from input length
		// Common cases: input is [inDim*batchSize], gradOut is [outDim]
		// We need to broadcast gradOut to [outDim, batchSize]
		if len(input.Data) > len(gradOutTensor.Data) && len(input.Data)%len(gradOutTensor.Data) == 0 {
			batchSize := len(input.Data) / len(gradOutTensor.Data)
			gradOutMat := tensor.New(len(gradOutTensor.Data), batchSize)
			for j := 0; j < len(gradOutTensor.Data); j++ {
				for b := 0; b < batchSize; b++ {
					gradOutMat.Data[j*batchSize+b] = gradOutTensor.Data[j]
				}
			}
			// Reshape input to [outDim, batchSize] for elementwise operation
			inputMat := tensor.New(len(gradOutTensor.Data), batchSize)
			copy(inputMat.Data, input.Data)
			// Now process elementwise
			gradIn := tensor.New(len(input.Data))
			c0 := 0.5
			c1 := 0.4244
			for i := range gradIn.Data {
				deriv := c0 + c1*input.Data[i]
				gradIn.Data[i] = gradOutMat.Data[i] * deriv
			}
			return gradIn, nil
		}
	}
	// Try to infer 2D shape
	rows, cols := 0, 0
	if len(input.Shape) == 2 && len(gradOutTensor.Shape) == 2 {
		rows, cols = input.Shape[0], input.Shape[1]
	} else if len(input.Data) > 0 && len(gradOutTensor.Data) > 0 {
		// Try to guess square-ish
		for try := 1; try*try <= len(input.Data); try++ {
			if try*try == len(input.Data) {
				rows, cols = try, try
				break
			}
		}
	}
	if rows > 0 && cols > 0 && rows*cols == len(input.Data) && len(gradOutTensor.Data) == len(input.Data) {
		gradIn := tensor.New(rows, cols)
		c0 := 0.5
		c1 := 0.4244
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				idx := i*cols + j
				deriv := c0 + c1*input.Data[idx]
				gradIn.Data[idx] = gradOutTensor.Data[idx] * deriv
			}
		}
		return gradIn, nil
	}
	// Fallback: error on shape mismatch
	return nil, fmt.Errorf("shape mismatch in Activation.Backward: input.Shape=%v, gradOut.Shape=%v", input.Shape, gradOutTensor.Shape)
}

// BackwardHE computes gradients for the encrypted (HE) case.
func (a *Activation) BackwardHE(gradOut *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if !a.encrypted {
		return nil, fmt.Errorf("BackwardHE called on plaintext activation layer")
	}
	if a.lastInput == nil {
		return nil, fmt.Errorf("No cached input for backward pass")
	}
	// Use the derivative polynomial (e.g., ReLU3_deriv)
	derivPoly, ok := SupportedPolynomials[a.poly.Name+"_deriv"]
	if !ok {
		return nil, fmt.Errorf("No derivative polynomial for %s", a.poly.Name)
	}
	// Evaluate derivative polynomial on lastInput
	derivCt, err := a.evalPolyOnCipher(a.lastInput, derivPoly)
	if err != nil {
		return nil, fmt.Errorf("Failed to evaluate derivative polynomial: %w", err)
	}
	// Multiply elementwise with gradOut
	res, err := a.serverKit.Evaluator.MulNew(gradOut, derivCt)
	if err != nil {
		return nil, fmt.Errorf("Failed to multiply gradOut and derivCt: %w", err)
	}
	res, err = a.serverKit.Evaluator.RelinearizeNew(res)
	if err != nil {
		return nil, err
	}
	if err = a.serverKit.Evaluator.Rescale(res, res); err != nil {
		return nil, err
	}
	return res, nil
}

// Forward processes the input through the layer.
func (a *Activation) Forward(input interface{}) (interface{}, error) {
	if a.encrypted {
		if ctInput, ok := input.(*rlwe.Ciphertext); ok {
			return a.ForwardCipher(ctInput)
		}
		if ctInputs, ok := input.([]*rlwe.Ciphertext); ok {
			out := make([]*rlwe.Ciphertext, len(ctInputs))
			for i, ct := range ctInputs {
				res, err := a.ForwardCipher(ct)
				if err != nil {
					return nil, err
				}
				out[i] = res
			}
			return out, nil
		}
		return nil, fmt.Errorf("encrypted activation expects *rlwe.Ciphertext or []*rlwe.Ciphertext input")
	}
	ptInput, ok := input.(*tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("plaintext activation expects *tensor.Tensor input")
	}
	return a.forwardPlain(ptInput)
}

// ForwardCipher processes an encrypted input through the layer.
func (a *Activation) ForwardCipher(input *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if !a.encrypted {
		return nil, fmt.Errorf("ForwardCipher expects encrypted activation layer")
	}
	// Cache input for backward
	a.lastInput = input.CopyNew()
	return a.forwardHE(input)
}

// forwardHE evaluates the polynomial on a ciphertext using Horner's method.
func (a *Activation) forwardHE(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if ct == nil {
		return nil, fmt.Errorf("input ciphertext is nil")
	}
	a.lastInput = ct // Cache input ciphertext for backward pass
	eval := a.serverKit.Evaluator
	coeffs := a.poly.Coeffs
	degree := a.poly.Degree

	// Start with the highest degree coefficient
	res, err := eval.AddNew(ct, 0) // copy of ct
	if err != nil {
		return nil, err
	}
	eval.Mul(res, 0, res) // zero it out

	// Encode coefficient as plaintext, matching ct's scale
	coeffVec := make([]complex128, a.heCtx.Params.MaxSlots())
	for i := range coeffVec {
		coeffVec[i] = complex(coeffs[degree], 0)
	}
	ptCoeff := ckks.NewPlaintext(a.heCtx.Params, res.Level())
	// Match scale to the ciphertext scale
	ptCoeff.Scale = ct.Scale
	a.heCtx.Encoder.Encode(coeffVec, ptCoeff)

	eval.Add(res, ptCoeff, res) // res = c_n

	for i := degree - 1; i >= 0; i-- {
		tmp, err := eval.MulNew(res, ct)
		if err != nil {
			return nil, err
		}
		tmp, err = eval.RelinearizeNew(tmp)
		if err != nil {
			return nil, err
		}
		if err = eval.Rescale(tmp, tmp); err != nil {
			return nil, err
		}
		if coeffs[i] != 0 {
			coeffVec := make([]complex128, a.heCtx.Params.MaxSlots())
			for j := range coeffVec {
				coeffVec[j] = complex(coeffs[i], 0)
			}
			ptCoeff := ckks.NewPlaintext(a.heCtx.Params, tmp.Level())
			// IMPORTANT: Match the scale to tmp's scale after rescaling!
			ptCoeff.Scale = tmp.Scale
			a.heCtx.Encoder.Encode(coeffVec, ptCoeff)
			res, err = eval.AddNew(tmp, ptCoeff)
			if err != nil {
				return nil, err
			}
		} else {
			res = tmp
		}
	}
	a.lastInput = ct.CopyNew()
	return res, nil
}

// forwardPlain evaluates the polynomial on a plaintext tensor.
func (a *Activation) forwardPlain(x *tensor.Tensor) (*tensor.Tensor, error) {
	a.lastInputShadow = tensor.New(len(x.Data))
	copy(a.lastInputShadow.Data, x.Data)
	y := tensor.New(x.Shape...)
	if a.poly.Name == "ReLU3" && !a.encrypted {
		for i, val := range x.Data {
			if val > 0 {
				y.Data[i] = val
			} else {
				y.Data[i] = 0
			}
		}
		return y, nil
	}
	coeffs := a.poly.Coeffs
	degree := a.poly.Degree
	for i, val := range x.Data {
		res := coeffs[degree]
		for j := degree - 1; j >= 0; j-- {
			res = res*val + coeffs[j]
		}
		y.Data[i] = res
	}
	return y, nil
}

// evalPolyOnCipher evaluates a polynomial on a ciphertext using Horner's method.
func (a *Activation) evalPolyOnCipher(ct *rlwe.Ciphertext, poly Poly) (*rlwe.Ciphertext, error) {
	eval := a.serverKit.Evaluator
	coeffs := poly.Coeffs
	degree := poly.Degree

	res, err := eval.AddNew(ct, 0)
	if err != nil {
		return nil, err
	}
	eval.Mul(res, 0, res)
	coeffVec := make([]complex128, a.heCtx.Params.MaxSlots())
	for i := range coeffVec {
		coeffVec[i] = complex(coeffs[degree], 0)
	}
	ptCoeff := ckks.NewPlaintext(a.heCtx.Params, res.Level())
	// Match scale to the input ciphertext scale
	ptCoeff.Scale = ct.Scale
	a.heCtx.Encoder.Encode(coeffVec, ptCoeff)
	eval.Add(res, ptCoeff, res)

	for i := degree - 1; i >= 0; i-- {
		tmp, err := eval.MulNew(res, ct)
		if err != nil {
			return nil, err
		}
		tmp, err = eval.RelinearizeNew(tmp)
		if err != nil {
			return nil, err
		}
		if err = eval.Rescale(tmp, tmp); err != nil {
			return nil, err
		}
		if coeffs[i] != 0 {
			coeffVec := make([]complex128, a.heCtx.Params.MaxSlots())
			for j := range coeffVec {
				coeffVec[j] = complex(coeffs[i], 0)
			}
			ptCoeff := ckks.NewPlaintext(a.heCtx.Params, tmp.Level())
			// IMPORTANT: Match the scale to tmp's scale after rescaling!
			ptCoeff.Scale = tmp.Scale
			a.heCtx.Encoder.Encode(coeffVec, ptCoeff)
			res, err = eval.AddNew(tmp, ptCoeff)
			if err != nil {
				return nil, err
			}
		} else {
			res = tmp
		}
	}
	return res, nil
}

func (a *Activation) Poly() Poly {
	return a.poly
}

func (a *Activation) HeContext() *ckkswrapper.HeContext {
	return a.heCtx
}

// Add interface methods for HE benchmarking
func (a *Activation) ForwardHE(x interface{}) (interface{}, error) {
	ct, ok := x.(*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected *rlwe.Ciphertext for HE input")
	}
	return a.forwardHE(ct)
}

// Interface method for HE backward (to avoid name conflict)
func (a *Activation) BackwardHEIface(g interface{}) (interface{}, error) {
	ct, ok := g.(*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected *rlwe.Ciphertext for HE grad")
	}
	return a.BackwardHE(ct)
}

func (a *Activation) Tag() string {
	return "Activation_" + a.poly.Name
}
