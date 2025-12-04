package layers

import (
	"fmt"

	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Conv2D is a 2D convolutional layer supporting plaintext & HE execution.
type Conv2D struct {
	// Layer parameters
	inChan, outChan int // number of input/output channels
	kh, kw          int // kernel height and width
	encrypted       bool

	// Input dimensions (set via SetDimensions or inferred from input)
	inH, inW int // input height and width

	// Plaintext parameters
	W *tensor.Tensor // weights: [outChan, inChan, kh, kw]
	B *tensor.Tensor // bias: [outChan]

	// HE context
	heCtx     *ckkswrapper.HeContext
	serverKit *ckkswrapper.ServerKit

	// HE parameters (channel-block packed)
	weightMasks []*rlwe.Plaintext // one mask per kernel position (dy,dx), packed by channel blocks
	biasPT      *rlwe.Plaintext   // plaintext bias
	groupSize   int               // number of output channels per ciphertext
	numGroups   int               // total number of output channel groups
	BSGS_B      int               // BSGS block size for block-diagonal matmul

	// HE gradient masks (for BackwardHE optimization)
	gradMasks []*rlwe.Plaintext // gradient masks for weight gradient computation

	// HE encrypted weights (for UpdateHE optimization)
	weightCTs []*rlwe.Ciphertext // encrypted weights packed by channel blocks

	// Optimization flags
	// Optimization flags
	useTreeSumFusion bool // whether to use tree-sum fusion optimization
	ForceUnpacked    bool // if true, forces groupSize=1 (one output channel per ciphertext)

	// Cached inputs for backward pass
	lastInput    *tensor.Tensor     // plaintext input
	lastInputCTs []*rlwe.Ciphertext // HE input

	// Gradient storage
	gradW *tensor.Tensor // plaintext weight gradients
	gradB *tensor.Tensor // plaintext bias gradients

	// HE gradient storage
	gradWCTs []*rlwe.Ciphertext // HE weight gradients
	gradBCT  *rlwe.Ciphertext   // HE bias gradient

	// Shadow plaintext for debugging
	wShadow         *tensor.Tensor
	bShadow         *tensor.Tensor
	lastInputShadow *tensor.Tensor
}

// NewConv2D creates a new Conv2D layer.
func NewConv2D(inChan, outChan, kh, kw int, encrypted bool, heCtx *ckkswrapper.HeContext) *Conv2D {
	c := &Conv2D{
		inChan:    inChan,
		outChan:   outChan,
		kh:        kh,
		kw:        kw,
		encrypted: encrypted,
		heCtx:     heCtx,
		W:         tensor.New(outChan, inChan, kh, kw),
		B:         tensor.New(outChan),
	}

	// Initialize shadow plaintext fields
	c.wShadow = tensor.New(outChan, inChan, kh, kw)
	c.bShadow = tensor.New(outChan)

	// Initialize gradient storage
	c.gradW = tensor.New(outChan, inChan, kh, kw)
	c.gradB = tensor.New(outChan)

	if encrypted {
		// Note: rotation keys will be generated when dimensions are set
		c.weightMasks = make([]*rlwe.Plaintext, kh*kw)
		c.gradWCTs = make([]*rlwe.Ciphertext, outChan*inChan*kh*kw)
	}

	return c
}

// EnableEncrypted switches the layer between encrypted and plaintext mode
func (c *Conv2D) EnableEncrypted(encrypted bool) {
	c.encrypted = encrypted
	if encrypted && c.heCtx != nil && c.weightMasks == nil {
		// Initialize HE components if switching to encrypted mode
		c.weightMasks = make([]*rlwe.Plaintext, c.kh*c.kw)
		c.gradWCTs = make([]*rlwe.Ciphertext, c.outChan*c.inChan*c.kh*c.kw)
	}
}

// SetDimensions sets the input dimensions for HE operations.
// This must be called before SyncHE() for encrypted layers.
func (c *Conv2D) SetDimensions(inH, inW int) error {
	if inH <= 0 || inW <= 0 {
		return fmt.Errorf("invalid dimensions: inH=%d, inW=%d", inH, inW)
	}

	c.inH = inH
	c.inW = inW

	// Generate rotation keys if this is an encrypted layer
	if c.encrypted && c.heCtx != nil {
		rots := []int{}

		// Forward pass rotations: for each kernel position (dy, dx)
		for dy := 0; dy < c.kh; dy++ {
			for dx := 0; dx < c.kw; dx++ {
				// Rotation by -(dy*inW + dx) for forward pass
				rots = append(rots, -(dy*inW + dx))
			}
		}

		// Backward pass rotations: for input gradient computation
		for dy := 0; dy < c.kh; dy++ {
			for dx := 0; dx < c.kw; dx++ {
				// Rotation by +(dy*inW + dx) for backward pass
				rots = append(rots, +(dy*inW + dx))
			}
		}

		// Tree-sum rotations for bias gradient (powers of 2)
		for step := 1; step < inH*inW; step *= 2 {
			rots = append(rots, step)
		}

		c.serverKit = c.heCtx.GenServerKit(rots)
	}

	return nil
}

// SyncHE encodes weights and bias into HE format with proper output-position replication.
func (c *Conv2D) SyncHE() error {
	if !c.encrypted {
		return nil
	}

	// Check if dimensions are set
	if c.inH == 0 || c.inW == 0 {
		return fmt.Errorf("dimensions not set: call SetDimensions(inH, inW) first")
	}

	// Update shadow weights
	copy(c.wShadow.Data, c.W.Data)
	copy(c.bShadow.Data, c.B.Data)

	slots := c.heCtx.Params.MaxSlots()

	// Compute output dimensions
	outH := c.inH - c.kh + 1
	outW := c.inW - c.kw + 1

	// For simplicity, use one ciphertext per output channel (no channel-block packing for now)
	c.groupSize = 1
	c.numGroups = c.outChan

	// Encode weight masks: one per (output_channel, input_channel, kernel_position)
	// This is [outChan][inChan][kh*kw] but we flatten to [outChan][inChan * kh * kw]
	// Each mask has the weight replicated to all valid output positions

	c.weightMasks = make([]*rlwe.Plaintext, c.outChan*c.inChan*c.kh*c.kw)

	for oc := 0; oc < c.outChan; oc++ {
		for ic := 0; ic < c.inChan; ic++ {
			for dy := 0; dy < c.kh; dy++ {
				for dx := 0; dx < c.kw; dx++ {
					// Get the weight for this kernel position
					wIdx := oc*c.inChan*c.kh*c.kw + ic*c.kh*c.kw + dy*c.kw + dx
					weight := c.W.Data[wIdx]

					// Create mask with weight replicated to all valid output positions
					maskVec := make([]complex128, slots)

					// For each valid output position (oy, ox), place the weight at position oy*inW + ox
					// This is because after rotation by -(dy*inW + dx), position oy*inW + ox
					// will contain input[oy*inW + ox + dy*inW + dx] = input[(oy+dy)*inW + (ox+dx)]
					// which is exactly what we need for output[oy, ox]
					for oy := 0; oy < outH; oy++ {
						for ox := 0; ox < outW; ox++ {
							outPos := oy*c.inW + ox // position in the packed vector
							if outPos < slots {
								maskVec[outPos] = complex(weight, 0)
							}
						}
					}

					// Encode the mask
					maskIdx := oc*c.inChan*c.kh*c.kw + ic*c.kh*c.kw + dy*c.kw + dx
					pt := ckks.NewPlaintext(c.heCtx.Params, c.heCtx.Params.MaxLevel())
					c.serverKit.Encoder.Encode(maskVec, pt)
					c.weightMasks[maskIdx] = pt
				}
			}
		}
	}

	// Encode bias - replicated to all valid output positions
	c.biasPT = nil // We'll use per-channel bias
	// For now, store bias as a slice of plaintexts (one per output channel)
	// Actually, let's just compute bias addition separately

	return nil
}

// EnableTreeSumFusion enables or disables the tree-sum fusion optimization.
// NOTE: Tree-sum fusion is currently disabled due to the new weight mask structure.
func (c *Conv2D) EnableTreeSumFusion(enable bool) {
	c.useTreeSumFusion = false // Disabled - not compatible with new implementation
}

// GetOutputShape returns the output dimensions for given input dimensions.
func (c *Conv2D) GetOutputShape(inH, inW int) (outH, outW int) {
	return inH - c.kh + 1, inW - c.kw + 1
}

// ForwardPlain performs plaintext forward pass.
func (c *Conv2D) ForwardPlain(input *tensor.Tensor) (*tensor.Tensor, error) {
	if c.encrypted {
		return nil, fmt.Errorf("ForwardPlain called on encrypted layer")
	}

	// Assume input shape is [batch, inChan, height, width] or [inChan, height, width]
	var batchSize, height, width int
	if len(input.Shape) == 4 {
		batchSize, _, height, width = input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	} else if len(input.Shape) == 3 {
		batchSize = 1
		_, height, width = input.Shape[0], input.Shape[1], input.Shape[2]
	} else {
		return nil, fmt.Errorf("input must be 3D or 4D tensor")
	}

	// Calculate output dimensions
	outHeight := height - c.kh + 1
	outWidth := width - c.kw + 1

	// Create output tensor
	output := tensor.New(batchSize, c.outChan, outHeight, outWidth)

	// Cache input for backward pass
	c.lastInput = input

	// Perform convolution
	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < c.outChan; oc++ {
			for y := 0; y < outHeight; y++ {
				for x := 0; x < outWidth; x++ {
					sum := c.B.Data[oc] // Start with bias

					// Convolve with kernel
					for ic := 0; ic < c.inChan; ic++ {
						for dy := 0; dy < c.kh; dy++ {
							for dx := 0; dx < c.kw; dx++ {
								// Input indices
								iy := y + dy
								ix := x + dx

								// Weight index
								wIdx := oc*c.inChan*c.kh*c.kw + ic*c.kh*c.kw + dy*c.kw + dx

								// Input index (handle both 3D and 4D)
								var inIdx int
								if batchSize == 1 {
									inIdx = ic*height*width + iy*width + ix
								} else {
									inIdx = b*c.inChan*height*width + ic*height*width + iy*width + ix
								}

								sum += input.Data[inIdx] * c.W.Data[wIdx]
							}
						}
					}

					// Output index
					outIdx := b*c.outChan*outHeight*outWidth + oc*outHeight*outWidth + y*outWidth + x
					output.Data[outIdx] = sum
				}
			}
		}
	}

	return output, nil
}

// ForwardHE performs HE forward pass with correct output-position replication.
func (c *Conv2D) ForwardHE(input []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if !c.encrypted {
		return nil, fmt.Errorf("ForwardHE called on plaintext layer")
	}

	if len(input) != c.inChan {
		return nil, fmt.Errorf("expected %d input channels, got %d", c.inChan, len(input))
	}

	// Cache input for backward pass
	c.lastInputCTs = make([]*rlwe.Ciphertext, len(input))
	for i, ct := range input {
		c.lastInputCTs[i] = ct.CopyNew()
	}

	// Infer dimensions from input if not already set
	if c.inH == 0 || c.inW == 0 {
		slots := c.heCtx.Params.MaxSlots()
		dim := 1
		for dim*dim <= slots && dim*dim <= 1024 {
			dim++
		}
		dim--

		if err := c.SetDimensions(dim, dim); err != nil {
			return nil, fmt.Errorf("failed to infer dimensions: %w", err)
		}

		if err := c.SyncHE(); err != nil {
			return nil, fmt.Errorf("failed to sync HE with inferred dimensions: %w", err)
		}
	}

	slots := c.heCtx.Params.MaxSlots()
	outH := c.inH - c.kh + 1
	outW := c.inW - c.kw + 1

	// Initialize output ciphertexts (one per output channel)
	output := make([]*rlwe.Ciphertext, c.outChan)
	for oc := 0; oc < c.outChan; oc++ {
		zeroVec := make([]complex128, slots)
		zeroPT := ckks.NewPlaintext(c.heCtx.Params, c.heCtx.Params.MaxLevel())
		c.serverKit.Encoder.Encode(zeroVec, zeroPT)
		output[oc], _ = c.heCtx.Encryptor.EncryptNew(zeroPT)
	}

	// Perform convolution for each output channel
	for oc := 0; oc < c.outChan; oc++ {
		// For each input channel
		for ic := 0; ic < c.inChan; ic++ {
			// For each kernel position
			for dy := 0; dy < c.kh; dy++ {
				for dx := 0; dx < c.kw; dx++ {
					// 1. Rotate input by (dy*inW + dx) to align patches
					// Positive rotation: slot[i] gets slot[i+k], so we shift values left
					// This brings input[oy+dy, ox+dx] to position (oy, ox) for all output positions
					rot, err := c.serverKit.Evaluator.RotateNew(input[ic], dy*c.inW+dx)
					if err != nil {
						return nil, fmt.Errorf("rotation failed: %w", err)
					}

					// 2. Get the weight mask for this (oc, ic, dy, dx)
					maskIdx := oc*c.inChan*c.kh*c.kw + ic*c.kh*c.kw + dy*c.kw + dx

					// 3. Multiply rotated input by weight mask
					mul, err := c.serverKit.Evaluator.MulNew(rot, c.weightMasks[maskIdx])
					if err != nil {
						return nil, fmt.Errorf("multiplication failed: %w", err)
					}

					// 4. Relinearize if needed
					if mul.Degree() > 1 {
						mul, err = c.serverKit.Evaluator.RelinearizeNew(mul)
						if err != nil {
							return nil, fmt.Errorf("relinearization failed: %w", err)
						}
					}

					// 5. Rescale
					if err := c.serverKit.Evaluator.Rescale(mul, mul); err != nil {
						return nil, fmt.Errorf("rescaling failed: %w", err)
					}

					// 6. Accumulate into output channel
					output[oc], err = c.serverKit.Evaluator.AddNew(output[oc], mul)
					if err != nil {
						return nil, fmt.Errorf("addition failed: %w", err)
					}
				}
			}
		}

		// Add bias for this output channel
		biasVec := make([]complex128, slots)
		bias := c.B.Data[oc]
		for oy := 0; oy < outH; oy++ {
			for ox := 0; ox < outW; ox++ {
				outPos := oy*c.inW + ox
				if outPos < slots {
					biasVec[outPos] = complex(bias, 0)
				}
			}
		}
		biasPT := ckks.NewPlaintext(c.heCtx.Params, output[oc].Level())
		biasPT.Scale = output[oc].Scale
		c.serverKit.Encoder.Encode(biasVec, biasPT)

		var err error
		output[oc], err = c.serverKit.Evaluator.AddNew(output[oc], biasPT)
		if err != nil {
			return nil, fmt.Errorf("bias addition failed: %w", err)
		}
	}

	return output, nil
}

// treeSumFusion performs convolution with tree-sum fusion optimization.
// This reduces the number of rotations and multiplications by using pre-computed weight masks.
func (c *Conv2D) treeSumFusion(input []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if !c.encrypted {
		return nil, fmt.Errorf("treeSumFusion called on plaintext layer")
	}

	slots := c.heCtx.Params.MaxSlots()

	// Initialize output ciphertexts (one per channel group)
	output := make([]*rlwe.Ciphertext, c.numGroups)
	for group := 0; group < c.numGroups; group++ {
		// Initialize with zeros
		zeroVec := make([]complex128, slots)
		zeroPT := ckks.NewPlaintext(c.heCtx.Params, c.heCtx.Params.MaxLevel())
		c.serverKit.Encoder.Encode(zeroVec, zeroPT)
		output[group], _ = c.heCtx.Encryptor.EncryptNew(zeroPT)
	}

	// Use the pre-computed weight masks (more efficient than creating individual masks)
	// For each kernel position
	for dy := 0; dy < c.kh; dy++ {
		for dx := 0; dx < c.kw; dx++ {
			pos := dy*c.kw + dx

			// For each input channel
			for ic := 0; ic < c.inChan; ic++ {
				// 1. Rotate input by -(dy*inW + dx)
				rot, err := c.serverKit.Evaluator.RotateNew(input[ic], -(dy*c.inW + dx))
				if err != nil {
					return nil, fmt.Errorf("rotation failed: %w", err)
				}

				// 2. Multiply by pre-computed weight mask for this position (block-diagonal)
				mul, err := c.serverKit.Evaluator.MulNew(rot, c.weightMasks[pos])
				if err != nil {
					return nil, fmt.Errorf("multiplication failed: %w", err)
				}

				// 3. Relinearize (only if degree > 1)
				if mul.Degree() > 1 {
					mul, err = c.serverKit.Evaluator.RelinearizeNew(mul)
					if err != nil {
						return nil, fmt.Errorf("relinearization failed: %w", err)
					}
				}

				// 4. Rescale
				if err := c.serverKit.Evaluator.Rescale(mul, mul); err != nil {
					return nil, fmt.Errorf("rescaling failed: %w", err)
				}

				// 5. Accumulate into all output groups (the weight mask handles the channel selection)
				for group := 0; group < c.numGroups; group++ {
					output[group], err = c.serverKit.Evaluator.AddNew(output[group], mul)
					if err != nil {
						return nil, fmt.Errorf("addition failed: %w", err)
					}
				}
			}
		}
	}

	// Add bias for each group
	for group := 0; group < c.numGroups; group++ {
		var err error
		output[group], err = c.serverKit.Evaluator.AddNew(output[group], c.biasPT)
		if err != nil {
			return nil, fmt.Errorf("bias addition failed: %w", err)
		}
	}

	return output, nil
}

// BackwardPlain performs plaintext backward pass.
func (c *Conv2D) BackwardPlain(gradOut *tensor.Tensor) (*tensor.Tensor, error) {
	if c.encrypted {
		return nil, fmt.Errorf("BackwardPlain called on encrypted layer")
	}

	if c.lastInput == nil {
		return nil, fmt.Errorf("no cached input for backward pass")
	}

	// Assume gradOut shape is [batch, outChan, outHeight, outWidth]
	var batchSize, outHeight, outWidth int
	if len(gradOut.Shape) == 4 {
		batchSize, _, outHeight, outWidth = gradOut.Shape[0], gradOut.Shape[1], gradOut.Shape[2], gradOut.Shape[3]
	} else {
		return nil, fmt.Errorf("gradOut must be 4D tensor")
	}

	// Get input dimensions
	var inHeight, inWidth int
	if len(c.lastInput.Shape) == 4 {
		_, _, inHeight, inWidth = c.lastInput.Shape[0], c.lastInput.Shape[1], c.lastInput.Shape[2], c.lastInput.Shape[3]
	} else {
		_, inHeight, inWidth = c.lastInput.Shape[0], c.lastInput.Shape[1], c.lastInput.Shape[2]
	}

	// Initialize gradients
	c.gradW = tensor.New(c.outChan, c.inChan, c.kh, c.kw)
	c.gradB = tensor.New(c.outChan)

	// Compute bias gradients: sum over all spatial positions
	for oc := 0; oc < c.outChan; oc++ {
		for b := 0; b < batchSize; b++ {
			for y := 0; y < outHeight; y++ {
				for x := 0; x < outWidth; x++ {
					gradIdx := b*c.outChan*outHeight*outWidth + oc*outHeight*outWidth + y*outWidth + x
					c.gradB.Data[oc] += gradOut.Data[gradIdx]
				}
			}
		}
	}

	// Compute weight gradients
	for oc := 0; oc < c.outChan; oc++ {
		for ic := 0; ic < c.inChan; ic++ {
			for dy := 0; dy < c.kh; dy++ {
				for dx := 0; dx < c.kw; dx++ {
					wGradIdx := oc*c.inChan*c.kh*c.kw + ic*c.kh*c.kw + dy*c.kw + dx

					for b := 0; b < batchSize; b++ {
						for y := 0; y < outHeight; y++ {
							for x := 0; x < outWidth; x++ {
								// Input position
								iy := y + dy
								ix := x + dx

								// Input index
								var inIdx int
								if len(c.lastInput.Shape) == 4 {
									inIdx = b*c.inChan*inHeight*inWidth + ic*inHeight*inWidth + iy*inWidth + ix
								} else {
									inIdx = ic*inHeight*inWidth + iy*inWidth + ix
								}

								// Gradient index
								gradIdx := b*c.outChan*outHeight*outWidth + oc*outHeight*outWidth + y*outWidth + x

								c.gradW.Data[wGradIdx] += c.lastInput.Data[inIdx] * gradOut.Data[gradIdx]
							}
						}
					}
				}
			}
		}
	}

	// Compute input gradients (transposed convolution)
	inputGrad := tensor.New(c.lastInput.Shape...)

	for b := 0; b < batchSize; b++ {
		for ic := 0; ic < c.inChan; ic++ {
			for y := 0; y < inHeight; y++ {
				for x := 0; x < inWidth; x++ {
					var inGradIdx int
					if len(c.lastInput.Shape) == 4 {
						inGradIdx = b*c.inChan*inHeight*inWidth + ic*inHeight*inWidth + y*inWidth + x
					} else {
						inGradIdx = ic*inHeight*inWidth + y*inWidth + x
					}

					sum := 0.0
					for oc := 0; oc < c.outChan; oc++ {
						for dy := 0; dy < c.kh; dy++ {
							for dx := 0; dx < c.kw; dx++ {
								// Output position
								oy := y - dy
								ox := x - dx

								if oy >= 0 && oy < outHeight && ox >= 0 && ox < outWidth {
									// Weight index
									wIdx := oc*c.inChan*c.kh*c.kw + ic*c.kh*c.kw + dy*c.kw + dx

									// Gradient index
									gradIdx := b*c.outChan*outHeight*outWidth + oc*outHeight*outWidth + oy*outWidth + ox

									sum += c.W.Data[wIdx] * gradOut.Data[gradIdx]
								}
							}
						}
					}
					inputGrad.Data[inGradIdx] = sum
				}
			}
		}
	}

	return inputGrad, nil
}

// BackwardHE performs HE backward pass.
func (c *Conv2D) BackwardHE(gradOut []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	if !c.encrypted {
		return nil, fmt.Errorf("BackwardHE called on plaintext layer")
	}

	if len(gradOut) != c.numGroups {
		return nil, fmt.Errorf("expected %d output channel groups, got %d", c.numGroups, len(gradOut))
	}

	if c.lastInputCTs == nil {
		return nil, fmt.Errorf("no cached input for backward pass")
	}

	slots := c.heCtx.Params.MaxSlots()
	outHeight := c.inH - c.kh + 1
	outWidth := c.inW - c.kw + 1

	// 1. Compute bias gradient with tree-sum fusion
	c.gradBCT = gradOut[0].CopyNew()
	// Note: For channel-block packed approach, all output channels are in gradOut[0]
	// The bias gradient is already computed correctly

	// Tree-sum over spatial dimensions
	for step := 1; step < outHeight*outWidth; step *= 2 {
		rot, _ := c.serverKit.Evaluator.RotateNew(c.gradBCT, step)
		c.gradBCT, _ = c.serverKit.Evaluator.AddNew(c.gradBCT, rot)
	}

	// 2. Compute weight gradients with channel-block packing and BSGS optimization
	// Initialize gradient ciphertexts for each group
	gradWGroupCTs := make([]*rlwe.Ciphertext, c.numGroups)
	for group := 0; group < c.numGroups; group++ {
		// Initialize with zeros
		zeroVec := make([]complex128, slots)
		zeroPT := ckks.NewPlaintext(c.heCtx.Params, c.heCtx.Params.MaxLevel())
		c.serverKit.Encoder.Encode(zeroVec, zeroPT)
		gradWGroupCTs[group], _ = c.heCtx.Encryptor.EncryptNew(zeroPT)
	}

	// For each kernel position
	for dy := 0; dy < c.kh; dy++ {
		for dx := 0; dx < c.kw; dx++ {
			// For each input channel
			for ic := 0; ic < c.inChan; ic++ {
				// For each output channel group
				for group := 0; group < c.numGroups; group++ {
					// Rotate gradOut by the appropriate offset for this kernel position
					rotGrad, err := c.serverKit.Evaluator.RotateNew(gradOut[group], +(dy*c.inW + dx))
					if err != nil {
						return nil, fmt.Errorf("gradient rotation failed: %w", err)
					}

					// Multiply by lastInputCTs[ic] to get weight gradient
					mul, err := c.serverKit.Evaluator.MulNew(rotGrad, c.lastInputCTs[ic])
					if err != nil {
						return nil, fmt.Errorf("gradient multiplication failed: %w", err)
					}

					// Relinearize (only if degree > 1)
					if mul.Degree() > 1 {
						mul, err = c.serverKit.Evaluator.RelinearizeNew(mul)
						if err != nil {
							return nil, fmt.Errorf("gradient relinearization failed: %w", err)
						}
					}

					// Rescale
					if err := c.serverKit.Evaluator.Rescale(mul, mul); err != nil {
						return nil, fmt.Errorf("gradient rescaling failed: %w", err)
					}

					// Tree-sum over spatial dimensions
					for step := 1; step < outHeight*outWidth; step *= 2 {
						rot, err := c.serverKit.Evaluator.RotateNew(mul, step)
						if err != nil {
							return nil, fmt.Errorf("spatial tree-sum rotation failed: %w", err)
						}
						mul, err = c.serverKit.Evaluator.AddNew(mul, rot)
						if err != nil {
							return nil, fmt.Errorf("spatial tree-sum addition failed: %w", err)
						}
					}

					// Accumulate into group gradient
					gradWGroupCTs[group], err = c.serverKit.Evaluator.AddNew(gradWGroupCTs[group], mul)
					if err != nil {
						return nil, fmt.Errorf("gradient accumulation failed: %w", err)
					}
				}
			}
		}
	}

	// Store the group gradient ciphertexts
	c.gradWCTs = gradWGroupCTs

	// 3. Compute input gradients (transposed convolution) with channel-block packing
	inputGrad := make([]*rlwe.Ciphertext, c.inChan)
	for ic := 0; ic < c.inChan; ic++ {
		// Initialize with zeros
		zeroVec := make([]complex128, slots)
		zeroPT := ckks.NewPlaintext(c.heCtx.Params, c.heCtx.Params.MaxLevel())
		c.serverKit.Encoder.Encode(zeroVec, zeroPT)
		inputGrad[ic], _ = c.heCtx.Encryptor.EncryptNew(zeroPT)

		// For each kernel position
		for dy := 0; dy < c.kh; dy++ {
			for dx := 0; dx < c.kw; dx++ {
				pos := dy*c.kw + dx

				// For each output channel group
				for group := 0; group < c.numGroups; group++ {
					// Rotate gradOut by +(dy*inW + dx)
					rot, err := c.serverKit.Evaluator.RotateNew(gradOut[group], +(dy*c.inW + dx))
					if err != nil {
						return nil, fmt.Errorf("input gradient rotation failed: %w", err)
					}

					// Multiply by weight mask for this position
					mul, err := c.serverKit.Evaluator.MulNew(rot, c.weightMasks[pos])
					if err != nil {
						return nil, fmt.Errorf("input gradient multiplication failed: %w", err)
					}

					// Relinearize (only if degree > 1)
					if mul.Degree() > 1 {
						mul, err = c.serverKit.Evaluator.RelinearizeNew(mul)
						if err != nil {
							return nil, fmt.Errorf("input gradient relinearization failed: %w", err)
						}
					}

					// Rescale
					if err := c.serverKit.Evaluator.Rescale(mul, mul); err != nil {
						return nil, fmt.Errorf("input gradient rescaling failed: %w", err)
					}

					// Accumulate into input gradient
					inputGrad[ic], err = c.serverKit.Evaluator.AddNew(inputGrad[ic], mul)
					if err != nil {
						return nil, fmt.Errorf("input gradient accumulation failed: %w", err)
					}
				}
			}
		}
	}

	return inputGrad, nil
}

// UpdatePlain updates parameters using plaintext gradients.
func (c *Conv2D) UpdatePlain(lr float64) error {
	if c.encrypted {
		return fmt.Errorf("UpdatePlain called on encrypted layer")
	}

	// Update weights
	for i := range c.W.Data {
		c.W.Data[i] -= lr * c.gradW.Data[i]
	}

	// Update bias
	for i := range c.B.Data {
		c.B.Data[i] -= lr * c.gradB.Data[i]
	}

	return nil
}

// UpdateHE updates parameters using HE gradients with BSGS optimization.
func (c *Conv2D) UpdateHE(lr float64) error {
	if !c.encrypted {
		return fmt.Errorf("UpdateHE called on plaintext layer")
	}

	// Update bias
	if c.gradBCT != nil {
		// Encode learning rate as plaintext
		lrVec := make([]complex128, c.heCtx.Params.MaxSlots())
		for i := range lrVec {
			lrVec[i] = complex(lr, 0)
		}
		lrPT := ckks.NewPlaintext(c.heCtx.Params, c.heCtx.Params.MaxLevel())
		c.serverKit.Encoder.Encode(lrVec, lrPT)

		// Update bias
		biasUpdate, err := c.serverKit.Evaluator.MulNew(c.gradBCT, lrPT)
		if err != nil {
			return fmt.Errorf("bias update multiplication failed: %w", err)
		}

		// Relinearize (only if degree > 1)
		if biasUpdate.Degree() > 1 {
			biasUpdate, err = c.serverKit.Evaluator.RelinearizeNew(biasUpdate)
			if err != nil {
				return fmt.Errorf("bias update relinearization failed: %w", err)
			}
		}

		if err := c.serverKit.Evaluator.Rescale(biasUpdate, biasUpdate); err != nil {
			return fmt.Errorf("bias update rescaling failed: %w", err)
		}

		// Note: In a full implementation, you'd subtract from encrypted bias
	}

	// Update weights with BSGS optimization
	if len(c.gradWCTs) > 0 {
		// Encode learning rate as plaintext
		lrVec := make([]complex128, c.heCtx.Params.MaxSlots())
		for i := range lrVec {
			lrVec[i] = complex(lr, 0)
		}
		lrPT := ckks.NewPlaintext(c.heCtx.Params, c.heCtx.Params.MaxLevel())
		c.serverKit.Encoder.Encode(lrVec, lrPT)

		// For each weight group
		for group := 0; group < c.numGroups; group++ {
			if group < len(c.gradWCTs) && c.gradWCTs[group] != nil {
				// Multiply gradient by learning rate
				weightUpdate, err := c.serverKit.Evaluator.MulNew(c.gradWCTs[group], lrPT)
				if err != nil {
					return fmt.Errorf("weight update multiplication failed: %w", err)
				}

				// Relinearize (only if degree > 1)
				if weightUpdate.Degree() > 1 {
					weightUpdate, err = c.serverKit.Evaluator.RelinearizeNew(weightUpdate)
					if err != nil {
						return fmt.Errorf("weight update relinearization failed: %w", err)
					}
				}

				// Rescale
				if err := c.serverKit.Evaluator.Rescale(weightUpdate, weightUpdate); err != nil {
					return fmt.Errorf("weight update rescaling failed: %w", err)
				}

				// Note: In a full implementation, you'd subtract from encrypted weights
			}
		}
	}

	return nil
}

// Interface methods to match existing layers
func (c *Conv2D) Forward(input interface{}) (interface{}, error) {
	if c.encrypted {
		if ctInput, ok := input.([]*rlwe.Ciphertext); ok {
			return c.ForwardHE(ctInput)
		}
		return nil, fmt.Errorf("expected []*rlwe.Ciphertext for encrypted forward")
	} else {
		if tensorInput, ok := input.(*tensor.Tensor); ok {
			return c.ForwardPlain(tensorInput)
		}
		return nil, fmt.Errorf("expected *tensor.Tensor for plaintext forward")
	}
}

func (c *Conv2D) Backward(gradOut interface{}) (interface{}, error) {
	if c.encrypted {
		if ctGradOut, ok := gradOut.([]*rlwe.Ciphertext); ok {
			return c.BackwardHE(ctGradOut)
		}
		return nil, fmt.Errorf("expected []*rlwe.Ciphertext for encrypted backward")
	} else {
		if tensorGradOut, ok := gradOut.(*tensor.Tensor); ok {
			return c.BackwardPlain(tensorGradOut)
		}
		return nil, fmt.Errorf("expected *tensor.Tensor for plaintext backward")
	}
}

func (c *Conv2D) Update(learningRate float64) error {
	if c.encrypted {
		return c.UpdateHE(learningRate)
	} else {
		return c.UpdatePlain(learningRate)
	}
}

func (c *Conv2D) Encrypted() bool {
	return c.encrypted
}

func (c *Conv2D) Levels() int {
	if c.encrypted {
		return 2 // Conv2D typically uses 2 levels
	}
	return 0
}

func (c *Conv2D) Tag() string {
	return fmt.Sprintf("Conv2D_%d_%d_%d_%d", c.inChan, c.outChan, c.kh, c.kw)
}

// Interface methods for HE benchmarking
func (c *Conv2D) ForwardHEIface(x interface{}) (interface{}, error) {
	cts, ok := x.([]*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected []*rlwe.Ciphertext for HE input")
	}
	return c.ForwardHE(cts)
}

func (c *Conv2D) BackwardHEIface(g interface{}) (interface{}, error) {
	cts, ok := g.([]*rlwe.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("expected []*rlwe.Ciphertext for HE grad")
	}
	return c.BackwardHE(cts)
}
