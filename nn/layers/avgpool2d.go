package layers

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"

	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type AvgPool2D struct {
	poolSize  int
	encrypted bool
	heCtx     *ckkswrapper.HeContext
	serverKit *ckkswrapper.ServerKit
	scalePT   *rlwe.Plaintext    // holds 1/p^2
	lastInput []*rlwe.Ciphertext // cache for HE backward
	// Add fields for input dimensions
	inH       int
	inW       int
	maskCache map[int]map[int]*rlwe.Plaintext // [level][idx] -> *Plaintext
}

func NewAvgPool2D(p int, encrypted bool, heCtx *ckkswrapper.HeContext) *AvgPool2D {
	layer := &AvgPool2D{
		poolSize:  p,
		encrypted: encrypted,
		heCtx:     heCtx,
	}
	if encrypted {
		offsets := make([]int, p*p-1)
		for i := 1; i < p*p; i++ {
			offsets[i-1] = i
		}
		layer.serverKit = heCtx.GenServerKit(offsets)
		// Pre-encode 1/(p^2) as plaintext
		inv := 1.0 / float64(p*p)
		vec := make([]complex128, heCtx.Params.MaxSlots())
		for i := range vec {
			vec[i] = complex(inv, 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		pt.Scale = heCtx.Params.DefaultScale()
		layer.serverKit.Encoder.Encode(vec, pt)
		layer.scalePT = pt
	}
	return layer
}

// EnableEncrypted switches the layer between encrypted and plaintext mode
func (a *AvgPool2D) EnableEncrypted(encrypted bool) {
	a.encrypted = encrypted
	if encrypted && a.heCtx != nil && a.serverKit == nil {
		// Initialize HE components if switching to encrypted mode
		offsets := make([]int, a.poolSize*a.poolSize-1)
		for i := 1; i < a.poolSize*a.poolSize; i++ {
			offsets[i-1] = i
		}
		a.serverKit = a.heCtx.GenServerKit(offsets)
		// Pre-encode 1/(p^2) as plaintext
		inv := 1.0 / float64(a.poolSize*a.poolSize)
		vec := make([]complex128, a.heCtx.Params.MaxSlots())
		for i := range vec {
			vec[i] = complex(inv, 0)
		}
		pt := ckks.NewPlaintext(a.heCtx.Params, a.heCtx.Params.MaxLevel())
		pt.Scale = a.heCtx.Params.DefaultScale()
		a.serverKit.Encoder.Encode(vec, pt)
		a.scalePT = pt
	}
}

func (a *AvgPool2D) ForwardPlain(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Input: [C,H,W] or [B,C,H,W]
	shape := x.Shape
	var B, C, H, W int
	if len(shape) == 3 {
		B, C, H, W = 1, shape[0], shape[1], shape[2]
	} else if len(shape) == 4 {
		B, C, H, W = shape[0], shape[1], shape[2], shape[3]
	} else {
		return nil, ErrType
	}
	p := a.poolSize
	outH, outW := H/p, W/p
	outShape := []int{C, outH, outW}
	if B > 1 {
		outShape = []int{B, C, outH, outW}
	}
	out := tensor.New(outShape...)
	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum := 0.0
					for ph := 0; ph < p; ph++ {
						for pw := 0; pw < p; pw++ {
							ih := oh*p + ph
							jw := ow*p + pw
							var idx int
							if B > 1 {
								idx = (((b*C+c)*H+ih)*W + jw)
							} else {
								idx = ((c*H+ih)*W + jw)
							}
							sum += x.Data[idx]
						}
					}
					avg := sum / float64(p*p)
					if B > 1 {
						out.Data[(((b*C+c)*outH+oh)*outW + ow)] = avg
					} else {
						out.Data[((c*outH+oh)*outW + ow)] = avg
					}
				}
			}
		}
	}
	return out, nil
}

func (a *AvgPool2D) ForwardHE(in []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	C := len(in)
	H, W, p := a.inH, a.inW, a.poolSize
	outH, outW := H/p, W/p
	Nslots := a.heCtx.Params.MaxSlots()

	out := make([]*rlwe.Ciphertext, C)
	zeroPT := ckks.NewPlaintext(a.heCtx.Params, a.heCtx.Params.MaxLevel())
	a.serverKit.Encoder.Encode(make([]complex128, Nslots), zeroPT)

	baseIdxs := make([]int, outH*outW)
	k := 0
	for by := 0; by < H; by += p {
		for bx := 0; bx < W; bx += p {
			baseIdxs[k] = by*W + bx
			k++
		}
	}

	for ch, c0 := range in {
		out[ch], _ = a.heCtx.Encryptor.EncryptNew(zeroPT)
		for k, base := range baseIdxs {
			sumCt, err := a.gatherSumSlot0(c0, base, W)
			if err != nil {
				return nil, err
			}
			avgCt, err := a.serverKit.Evaluator.MulNew(sumCt, a.scalePT)
			if err != nil {
				return nil, err
			}
			if err := a.serverKit.Evaluator.Rescale(avgCt, avgCt); err != nil {
				return nil, err
			}
			// Mask slot 0 to prevent value leakage on rotation
			maskPT := a.oneHotPlain(0, avgCt.Level(), avgCt.Scale)
			avgMasked, err := a.serverKit.Evaluator.MulNew(avgCt, maskPT)
			if err != nil {
				return nil, err
			}
			if avgMasked.Degree() > 1 {
				_ = a.serverKit.Evaluator.Relinearize(avgMasked, avgMasked)
			}
			_ = a.serverKit.Evaluator.Rescale(avgMasked, avgMasked)
			// Rotate -k to place the average in slot k (see sign convention below)
			placed, err := a.serverKit.Evaluator.RotateNew(avgMasked, -k)
			if err != nil {
				return nil, err
			}
			out[ch], err = a.serverKit.Evaluator.AddNew(out[ch], placed)
			if err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func (a *AvgPool2D) BackwardHE(dOut []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if len(dOut) == 0 {
		return nil, fmt.Errorf("input gradient slice is empty")
	}

	C := len(dOut)
	p, H, W := a.poolSize, a.inH, a.inW
	eval := a.serverKit.Evaluator
	level := dOut[0].Level()

	// Initialize output gradients with encrypted zeros at the same level as dOut
	dIn := make([]*rlwe.Ciphertext, C)
	zeroPT := ckks.NewPlaintext(a.heCtx.Params, level)
	a.serverKit.Encoder.Encode(make([]complex128, a.heCtx.Params.MaxSlots()), zeroPT)
	for i := 0; i < C; i++ {
		dIn[i], _ = a.heCtx.Encryptor.EncryptNew(zeroPT)
	}

	// Compute base indices (top-left corners) for each pooling window
	outH, outW := H/p, W/p
	baseIdxs := make([]int, outH*outW)
	k := 0
	for by := 0; by < H; by += p {
		for bx := 0; bx < W; bx += p {
			baseIdxs[k] = by*W + bx
			k++
		}
	}

	// Scatter gradients: for each output slot k, distribute to its p*p input slots
	for ch, gCh := range dOut {
		for k, base := range baseIdxs {
			// Isolate gradient at slot k
			maskK := a.oneHotPlain(k, gCh.Level(), gCh.Scale)
			isolatedGrad, err := eval.MulNew(gCh, maskK)
			if err != nil {
				return nil, err
			}
			if isolatedGrad.Degree() > 1 {
				_ = eval.Relinearize(isolatedGrad, isolatedGrad)
			}
			if err := eval.Rescale(isolatedGrad, isolatedGrad); err != nil {
				return nil, err
			}

			// Scale by 1/(p*p)
			scaledGrad, err := eval.MulNew(isolatedGrad, a.scalePT)
			if err != nil {
				return nil, err
			}
			if scaledGrad.Degree() > 1 {
				_ = eval.Relinearize(scaledGrad, scaledGrad)
			}
			if err := eval.Rescale(scaledGrad, scaledGrad); err != nil {
				return nil, err
			}

			// Distribute to each position in the p x p window
			for dy := 0; dy < p; dy++ {
				for dx := 0; dx < p; dx++ {
					targetIdx := base + dy*W + dx
					rot := k - targetIdx
					term, err := eval.RotateNew(scaledGrad, rot)
					if err != nil {
						return nil, fmt.Errorf("rotation failed for targetIdx %d: %w", targetIdx, err)
					}
					var addErr error
					dIn[ch], addErr = eval.AddNew(dIn[ch], term)
					if addErr != nil {
						return nil, addErr
					}
				}
			}
		}
	}

	return dIn, nil
}

func (a *AvgPool2D) UpdateHE(lr float64) error {
	// No-op for pooling
	return nil
}

// Add SetDimensions method
func (a *AvgPool2D) SetDimensions(inH, inW int) error {
	a.inH = inH
	a.inW = inW
	if a.encrypted && a.heCtx != nil {
		p := a.poolSize
		H, W := inH, inW
		rotSet := make(map[int]struct{})
		k := 0
		for by := 0; by < H; by += p {
			for bx := 0; bx < W; bx += p {
				base := by*W + bx
				rotSet[k] = struct{}{}  // +k for placement
				rotSet[-k] = struct{}{} // -k for placement (needed for placement rotation)
				for dy := 0; dy < p; dy++ {
					for dx := 0; dx < p; dx++ {
						idx := base + dy*W + dx
						rotSet[idx] = struct{}{}  // +idx for each pixel
						rotSet[-idx] = struct{}{} // -idx for backward scatter
					}
				}
				k++
			}
		}
		rotList := make([]int, 0, len(rotSet))
		for rot := range rotSet {
			if rot != 0 {
				rotList = append(rotList, rot)
			}
		}
		a.serverKit = a.heCtx.GenServerKit(rotList)
	}
	return nil
}

// Add SyncHE method (stub)
func (a *AvgPool2D) SyncHE() error {
	if !a.encrypted {
		return nil
	}
	if a.inH == 0 || a.inW == 0 {
		return fmt.Errorf("dimensions not set: call SetDimensions(inH, inW) first")
	}
	return nil
}

// Update Levels() docstring to reflect 1 level for HE AvgPool2D.
func (a *AvgPool2D) Levels() int {
	// AvgPool2D (HE) uses only 1 level due to masking-based pooling.
	return 1
}

// gatherSum returns a ciphertext where every slot holds the sum of the p x p window starting at base
func (a *AvgPool2D) gatherSum(ct *rlwe.Ciphertext, base, W int) (*rlwe.Ciphertext, error) {
	slots := a.heCtx.Params.MaxSlots()
	encoder := a.serverKit.Encoder
	eval := a.serverKit.Evaluator

	// zero-initialised ciphertext to accumulate into
	zeroPT := ckks.NewPlaintext(a.heCtx.Params, ct.Level())
	acc, _ := a.heCtx.Encryptor.EncryptNew(zeroPT)

	for dy := 0; dy < a.poolSize; dy++ {
		for dx := 0; dx < a.poolSize; dx++ {
			idx := base + dy*W + dx
			if idx >= slots {
				continue
			}
			maskVec := make([]complex128, slots)
			maskVec[idx] = 1
			ptMask := ckks.NewPlaintext(a.heCtx.Params, ct.Level())
			ptMask.Scale = ct.Scale
			encoder.Encode(maskVec, ptMask)

			term, err := eval.MulNew(ct, ptMask)
			if err != nil {
				return nil, err
			}
			if term.Degree() > 1 {
				_ = eval.Relinearize(term, term)
			}
			if err := eval.Rescale(term, term); err != nil {
				return nil, err
			}

			acc, err = eval.AddNew(acc, term)
			if err != nil {
				return nil, err
			}
		}
	}
	return acc, nil // every slot now holds the window-sum
}

// isolateSlot0 masks all slots except slot 0
func isolateSlot0(sumCt *rlwe.Ciphertext, heCtx *ckkswrapper.HeContext, eval *ckks.Evaluator, encoder ckks.Encoder) (*rlwe.Ciphertext, error) {
	slots := heCtx.Params.MaxSlots()
	mask := make([]complex128, slots)
	mask[0] = 1
	ptMask := ckks.NewPlaintext(heCtx.Params, sumCt.Level())
	ptMask.Scale = sumCt.Scale
	encoder.Encode(mask, ptMask)
	iso, err := eval.MulNew(sumCt, ptMask)
	if err != nil {
		return nil, err
	}
	if iso.Degree() > 1 {
		_ = eval.Relinearize(iso, iso)
	}
	return iso, eval.Rescale(iso, iso)
}

// Helper: one-hot plaintext cache
func (a *AvgPool2D) oneHotPlain(idx int, level int, scale rlwe.Scale) *rlwe.Plaintext {
	if a.maskCache == nil {
		a.maskCache = make(map[int]map[int]*rlwe.Plaintext)
	}
	if _, ok := a.maskCache[level]; !ok {
		a.maskCache[level] = make(map[int]*rlwe.Plaintext)
	}
	if pt, ok := a.maskCache[level][idx]; ok {
		return pt
	}
	vec := make([]complex128, a.heCtx.Params.MaxSlots())
	vec[idx] = 1
	pt := ckks.NewPlaintext(a.heCtx.Params, level)
	pt.Scale = scale
	a.heCtx.Encoder.Encode(vec, pt)
	a.maskCache[level][idx] = pt
	return pt
}

// gatherSumSlot0: sum a p x p window (top-left slot = base), return ct with sum in slot 0
func (a *AvgPool2D) gatherSumSlot0(ct *rlwe.Ciphertext, base, W int) (*rlwe.Ciphertext, error) {
	slots := a.heCtx.Params.MaxSlots()
	eval := a.serverKit.Evaluator
	var sumCt *rlwe.Ciphertext
	for dy := 0; dy < a.poolSize; dy++ {
		for dx := 0; dx < a.poolSize; dx++ {
			idx := base + dy*W + dx
			if idx >= slots {
				continue
			}
			ptMask := a.oneHotPlain(idx, ct.Level(), ct.Scale)
			isolated, err := eval.MulNew(ct, ptMask)
			if err != nil {
				return nil, err
			}
			if isolated.Degree() > 1 {
				_ = eval.Relinearize(isolated, isolated)
			}
			_ = eval.Rescale(isolated, isolated)
			// Use Rotate(+idx) to bring slot idx to slot 0 (gather phase)
			rotIso, err := eval.RotateNew(isolated, +idx)
			if err != nil {
				return nil, err
			}
			if sumCt == nil {
				sumCt = rotIso.CopyNew()
			} else {
				sumCt, err = eval.AddNew(sumCt, rotIso)
				if err != nil {
					return nil, err
				}
			}
		}
	}
	return sumCt, nil
}

// Interface methods for HE benchmarking
func (a *AvgPool2D) ForwardHEIface(x interface{}) (interface{}, error) {
	cts, ok := x.([]*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected []*rlwe.Ciphertext for HE input")
	}
	return a.ForwardHE(cts)
}

func (a *AvgPool2D) BackwardHEIface(g interface{}) (interface{}, error) {
	cts, ok := g.([]*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected []*rlwe.Ciphertext for HE grad")
	}
	return a.BackwardHE(cts)
}

func (a *AvgPool2D) Tag() string {
	return fmt.Sprintf("AvgPool2D_%d", a.poolSize)
}
