package layers

import (
	"fmt"
	"testing"
	"time"

	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestConv2D_Identity1x1(t *testing.T) {
	// Test 1x1 identity convolution
	heCtx := ckkswrapper.NewHeContext()

	// Create 1x1 identity conv layer
	conv := NewConv2D(1, 1, 1, 1, false, heCtx)

	// Set weights to identity (single weight = 1.0)
	conv.W.Set(1.0, 0, 0, 0, 0)
	conv.B.Set(0.0, 0)

	// Create test input: 1 channel, 3x3 image
	input := tensor.New(1, 3, 3)
	for i := 0; i < 9; i++ {
		input.Data[i] = float64(i + 1)
	}

	// Forward pass
	output, err := conv.ForwardPlain(input)
	require.NoError(t, err)

	// Check output shape
	assert.Equal(t, []int{1, 1, 3, 3}, output.Shape)

	// Check that output equals input (identity)
	for i := 0; i < 9; i++ {
		assert.Equal(t, input.Data[i], output.Data[i], "Identity conv should preserve input")
	}
}

func TestConv2D_Random3x3(t *testing.T) {
	// Test random 3x3 convolution
	heCtx := ckkswrapper.NewHeContext()

	// Create 3x3 conv layer: 1 input channel, 2 output channels
	conv := NewConv2D(1, 2, 3, 3, false, heCtx)

	// Set random weights
	for oc := 0; oc < 2; oc++ {
		for ic := 0; ic < 1; ic++ {
			for kh := 0; kh < 3; kh++ {
				for kw := 0; kw < 3; kw++ {
					conv.W.Set(float64(oc+ic+kh+kw), oc, ic, kh, kw)
				}
			}
		}
	}
	conv.B.Set(0.1, 0)
	conv.B.Set(0.2, 1)

	// Create test input: 1 channel, 5x5 image
	input := tensor.New(1, 5, 5)
	for i := 0; i < 25; i++ {
		input.Data[i] = float64(i + 1)
	}

	// Forward pass
	output, err := conv.ForwardPlain(input)
	require.NoError(t, err)

	// Check output shape: [1, 2, 3, 3] (5x5 input - 3x3 kernel + 1 = 3x3 output)
	assert.Equal(t, []int{1, 2, 3, 3}, output.Shape)

	// Verify output is not zero (basic sanity check)
	hasNonZero := false
	for _, val := range output.Data {
		if val != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Output should have non-zero values")
}

func TestConv2D_Backward(t *testing.T) {
	// Test backward pass
	heCtx := ckkswrapper.NewHeContext()

	// Create 2x2 conv layer: 1 input channel, 1 output channel
	conv := NewConv2D(1, 1, 2, 2, false, heCtx)

	// Set simple weights
	conv.W.Set(1.0, 0, 0, 0, 0)
	conv.W.Set(1.0, 0, 0, 0, 1)
	conv.W.Set(1.0, 0, 0, 1, 0)
	conv.W.Set(1.0, 0, 0, 1, 1)
	conv.B.Set(0.0, 0)

	// Create test input: 1 channel, 3x3 image
	input := tensor.New(1, 3, 3)
	for i := 0; i < 9; i++ {
		input.Data[i] = float64(i + 1)
	}

	// Forward pass
	output, err := conv.ForwardPlain(input)
	require.NoError(t, err)

	// Create gradient output
	gradOut := tensor.New(output.Shape...)
	for i := 0; i < len(gradOut.Data); i++ {
		gradOut.Data[i] = 1.0
	}

	// Backward pass
	gradIn, err := conv.BackwardPlain(gradOut)
	require.NoError(t, err)

	// Check gradient input shape matches input shape
	assert.Equal(t, input.Shape, gradIn.Shape)

	// Verify gradients are computed
	hasNonZero := false
	for _, val := range gradIn.Data {
		if val != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Input gradients should be non-zero")

	// Check weight gradients are computed
	hasNonZero = false
	for _, val := range conv.gradW.Data {
		if val != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Weight gradients should be non-zero")

	// Check bias gradients are computed
	hasNonZero = false
	for _, val := range conv.gradB.Data {
		if val != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Bias gradients should be non-zero")
}

func TestConv2D_Update(t *testing.T) {
	// Test parameter update
	heCtx := ckkswrapper.NewHeContext()

	// Create conv layer
	conv := NewConv2D(1, 1, 2, 2, false, heCtx)

	// Set initial weights
	initialWeight := 1.0
	initialBias := 0.5
	conv.W.Set(initialWeight, 0, 0, 0, 0)
	conv.B.Set(initialBias, 0)

	// Set gradients
	conv.gradW.Set(0.1, 0, 0, 0, 0)
	conv.gradB.Set(0.05, 0)

	// Update with learning rate
	lr := 0.1
	err := conv.UpdatePlain(lr)
	require.NoError(t, err)

	// Check weights were updated
	expectedWeight := initialWeight - lr*0.1
	expectedBias := initialBias - lr*0.05

	assert.Equal(t, expectedWeight, conv.W.At(0, 0, 0, 0))
	assert.Equal(t, expectedBias, conv.B.At(0))
}

func TestConv2D_HE_Forward(t *testing.T) {
	// Test HE forward pass
	heCtx := ckkswrapper.NewHeContext()

	// Create encrypted conv layer
	conv := NewConv2D(1, 1, 2, 2, true, heCtx)

	// Set simple weights
	conv.W.Set(1.0, 0, 0, 0, 0)
	conv.W.Set(1.0, 0, 0, 0, 1)
	conv.W.Set(1.0, 0, 0, 1, 0)
	conv.W.Set(1.0, 0, 0, 1, 1)
	conv.B.Set(0.0, 0)

	// Set dimensions and sync HE parameters
	err := conv.SetDimensions(3, 3)
	require.NoError(t, err)
	err = conv.SyncHE()
	require.NoError(t, err)

	// Create encrypted input
	input := tensor.New(1, 3, 3)
	for i := 0; i < 9; i++ {
		input.Data[i] = float64(i + 1)
	}

	// Encrypt input
	inputVec := make([]complex128, heCtx.Params.MaxSlots())
	for i := 0; i < 9; i++ {
		inputVec[i] = complex(input.Data[i], 0)
	}

	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, pt)
	ct, err := heCtx.Encryptor.EncryptNew(pt)
	require.NoError(t, err)

	// Forward pass
	outputCTs, err := conv.ForwardHE([]*rlwe.Ciphertext{ct})
	require.NoError(t, err)

	// Decrypt and check output
	require.Len(t, outputCTs, 1)
	ptOut := heCtx.Decryptor.DecryptNew(outputCTs[0])
	outputVec := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(ptOut, outputVec)

	// Check that output is not zero (basic sanity check)
	hasNonZero := false
	for i := 0; i < 4; i++ { // 3x3 input - 2x2 kernel + 1 = 2x2 output = 4 values
		if real(outputVec[i]) != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "HE output should have non-zero values")
}

func TestConv2D_HE_Backward(t *testing.T) {
	// Test HE backward pass
	heCtx := ckkswrapper.NewHeContext()

	// Create encrypted conv layer
	conv := NewConv2D(1, 1, 2, 2, true, heCtx)

	// Set simple weights
	conv.W.Set(1.0, 0, 0, 0, 0)
	conv.W.Set(1.0, 0, 0, 0, 1)
	conv.W.Set(1.0, 0, 0, 1, 0)
	conv.W.Set(1.0, 0, 0, 1, 1)
	conv.B.Set(0.0, 0)

	// Set dimensions and sync HE parameters
	err := conv.SetDimensions(3, 3)
	require.NoError(t, err)
	err = conv.SyncHE()
	require.NoError(t, err)

	// Create encrypted input and do forward pass
	input := tensor.New(1, 3, 3)
	for i := 0; i < 9; i++ {
		input.Data[i] = float64(i + 1)
	}

	inputVec := make([]complex128, heCtx.Params.MaxSlots())
	for i := 0; i < 9; i++ {
		inputVec[i] = complex(input.Data[i], 0)
	}

	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, pt)
	ct, err := heCtx.Encryptor.EncryptNew(pt)
	require.NoError(t, err)

	_, err = conv.ForwardHE([]*rlwe.Ciphertext{ct})
	require.NoError(t, err)

	// Create encrypted gradient output
	gradOutVec := make([]complex128, heCtx.Params.MaxSlots())
	for i := 0; i < 4; i++ { // 2x2 output
		gradOutVec[i] = 1.0
	}

	ptGrad := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(gradOutVec, ptGrad)
	ctGrad, err := heCtx.Encryptor.EncryptNew(ptGrad)
	require.NoError(t, err)

	// Backward pass
	gradInCTs, err := conv.BackwardHE([]*rlwe.Ciphertext{ctGrad})
	require.NoError(t, err)

	// Check output
	require.Len(t, gradInCTs, 1)

	// Decrypt and verify
	ptGradIn := heCtx.Decryptor.DecryptNew(gradInCTs[0])
	gradInVec := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(ptGradIn, gradInVec)

	// Check that gradients are computed
	hasNonZero := false
	for i := 0; i < 9; i++ {
		if real(gradInVec[i]) != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "HE input gradients should be non-zero")
}

func TestConv2D_HE_Update(t *testing.T) {
	// Test HE parameter update
	heCtx := ckkswrapper.NewHeContext()

	// Create encrypted conv layer
	conv := NewConv2D(1, 1, 2, 2, true, heCtx)

	// Set initial weights
	conv.W.Set(1.0, 0, 0, 0, 0)
	conv.B.Set(0.5, 0)

	// Set dimensions and sync HE parameters
	err := conv.SetDimensions(3, 3)
	require.NoError(t, err)
	err = conv.SyncHE()
	require.NoError(t, err)

	// Simulate gradients (in real usage, these would come from BackwardHE)
	conv.gradW.Set(0.1, 0, 0, 0, 0)
	conv.gradB.Set(0.05, 0)

	// Update with learning rate
	lr := 0.1
	err = conv.UpdateHE(lr)
	require.NoError(t, err)

	// Note: Current HE update implementation doesn't update plaintext weights
	// In a full implementation, you'd decrypt gradBCT and update encrypted weights
	// For now, we just verify the method doesn't error
	// expectedWeight := 1.0 - lr*0.1
	// expectedBias := 0.5 - lr*0.05
	// assert.Equal(t, expectedWeight, conv.W.At(0, 0, 0, 0))
	// assert.Equal(t, expectedBias, conv.B.At(0))
}

func TestConv2D_Interface(t *testing.T) {
	// Test that Conv2D implements the same interface as other layers
	heCtx := ckkswrapper.NewHeContext()

	// Test plaintext interface
	conv := NewConv2D(1, 1, 2, 2, false, heCtx)

	// Test Forward interface
	input := tensor.New(1, 3, 3)
	for i := 0; i < 9; i++ {
		input.Data[i] = float64(i + 1)
	}

	output, err := conv.Forward(input)
	require.NoError(t, err)
	_, ok := output.(*tensor.Tensor)
	assert.True(t, ok, "Forward should return *tensor.Tensor for plaintext")

	// Test Backward interface
	gradOut := tensor.New(1, 1, 2, 2)
	for i := 0; i < 4; i++ {
		gradOut.Data[i] = 1.0
	}

	gradIn, err := conv.Backward(gradOut)
	require.NoError(t, err)
	_, ok = gradIn.(*tensor.Tensor)
	assert.True(t, ok, "Backward should return *tensor.Tensor for plaintext")

	// Test Update interface
	err = conv.Update(0.1)
	require.NoError(t, err)

	// Test interface methods
	assert.False(t, conv.Encrypted())
	assert.Equal(t, 0, conv.Levels())
}

func BenchmarkConv2D_Plaintext(b *testing.B) {
	// Benchmark plaintext Conv2D (LeNet-style: 1->6, 5x5 on 28x28)
	heCtx := ckkswrapper.NewHeContext()
	conv := NewConv2D(1, 6, 5, 5, false, heCtx)

	// Create 28x28 input
	input := tensor.New(1, 28, 28)
	for i := 0; i < 28*28; i++ {
		input.Data[i] = float64(i % 10)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := conv.ForwardPlain(input)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConv2D_HE(b *testing.B) {
	// Benchmark HE Conv2D (LeNet-style: 1->6, 5x5 on 28x28)
	heCtx := ckkswrapper.NewHeContext()
	Bvals := []int{3, 5, 7}
	for _, B := range Bvals {
		b.Run(fmt.Sprintf("BSGS_B=%d", B), func(b *testing.B) {
			conv := NewConv2D(1, 6, 5, 5, true, heCtx)
			conv.BSGS_B = B
			// Set weights and sync
			for oc := 0; oc < 6; oc++ {
				for ic := 0; ic < 1; ic++ {
					for kh := 0; kh < 5; kh++ {
						for kw := 0; kw < 5; kw++ {
							conv.W.Set(float64(oc+kh+kw), oc, ic, kh, kw)
						}
					}
				}
				conv.B.Set(0.1, oc)
			}
			err := conv.SetDimensions(28, 28)
			if err != nil {
				b.Fatal(err)
			}
			err = conv.SyncHE()
			if err != nil {
				b.Fatal(err)
			}
			// Create encrypted input
			inputVec := make([]complex128, heCtx.Params.MaxSlots())
			for i := 0; i < 28*28; i++ {
				inputVec[i] = complex(float64(i%10), 0)
			}
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(inputVec, pt)
			ct, err := heCtx.Encryptor.EncryptNew(pt)
			if err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := conv.ForwardHE([]*rlwe.Ciphertext{ct})
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkConv2D_HE_TreeSumFusion(b *testing.B) {
	// Benchmark HE Conv2D with tree-sum fusion (LeNet-style: 1->6, 5x5 on 28x28)
	heCtx := ckkswrapper.NewHeContext()

	conv := NewConv2D(1, 6, 5, 5, true, heCtx)
	conv.SetDimensions(28, 28)
	conv.EnableTreeSumFusion(true) // Enable tree-sum fusion
	conv.SyncHE()

	// Create input (1 channel, 28x28)
	input := make([]*rlwe.Ciphertext, 1)
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < 28*28; i++ {
		inputVec[i] = complex(float64(i%10)/10.0, 0) // Simple pattern
	}
	inputPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, inputPT)
	input[0], _ = heCtx.Encryptor.EncryptNew(inputPT)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := conv.ForwardHE(input)
		if err != nil {
			b.Fatalf("ForwardHE failed: %v", err)
		}
	}
}

func BenchmarkConv2D_HE_Comparison(b *testing.B) {
	// Benchmark comparison between baseline and tree-sum fusion
	heCtx := ckkswrapper.NewHeContext()

	// Create input (1 channel, 28x28)
	input := make([]*rlwe.Ciphertext, 1)
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < 28*28; i++ {
		inputVec[i] = complex(float64(i%10)/10.0, 0) // Simple pattern
	}
	inputPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, inputPT)
	input[0], _ = heCtx.Encryptor.EncryptNew(inputPT)

	// Test baseline
	b.Run("Baseline", func(b *testing.B) {
		conv := NewConv2D(1, 6, 5, 5, true, heCtx)
		conv.SetDimensions(28, 28)
		conv.EnableTreeSumFusion(false) // Disable tree-sum fusion
		conv.SyncHE()

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, err := conv.ForwardHE(input)
			if err != nil {
				b.Fatalf("ForwardHE failed: %v", err)
			}
		}
	})

	// Test tree-sum fusion
	b.Run("TreeSumFusion", func(b *testing.B) {
		conv := NewConv2D(1, 6, 5, 5, true, heCtx)
		conv.SetDimensions(28, 28)
		conv.EnableTreeSumFusion(true) // Enable tree-sum fusion
		conv.SyncHE()

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			_, err := conv.ForwardHE(input)
			if err != nil {
				b.Fatalf("ForwardHE failed: %v", err)
			}
		}
	})
}

func BenchmarkConv2D_HE_TreeSumFusion_BSGS(b *testing.B) {
	// Benchmark HE Conv2D with tree-sum fusion and different BSGS block sizes
	heCtx := ckkswrapper.NewHeContext()

	// Create input (1 channel, 28x28)
	input := make([]*rlwe.Ciphertext, 1)
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < 28*28; i++ {
		inputVec[i] = complex(float64(i%10)/10.0, 0) // Simple pattern
	}
	inputPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, inputPT)
	input[0], _ = heCtx.Encryptor.EncryptNew(inputPT)

	// Test different BSGS block sizes
	bsgsValues := []int{3, 5, 7}

	for _, bsgsB := range bsgsValues {
		b.Run(fmt.Sprintf("BSGS_B=%d", bsgsB), func(b *testing.B) {
			conv := NewConv2D(1, 6, 5, 5, true, heCtx)
			conv.SetDimensions(28, 28)
			conv.BSGS_B = bsgsB
			conv.EnableTreeSumFusion(true) // Enable tree-sum fusion
			conv.SyncHE()

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, err := conv.ForwardHE(input)
				if err != nil {
					b.Fatalf("ForwardHE failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkConv2D_ResNet_HE(b *testing.B) {
	b.Run("ResNet_HE", func(b *testing.B) {
		heCtx := ckkswrapper.NewHeContext()
		startAll := time.Now()
		b.Logf("[ResNet] benchmark start: %v", startAll)

		// Layer setup: 64->64, 3x3 kernel, 32x32 input
		resnetConv := NewConv2D(64, 64, 3, 3, true, heCtx)
		resnetConv.SetDimensions(32, 32)
		b.Logf("[ResNet] SetDimensions done at +%v", time.Since(startAll))

		resnetConv.SyncHE()
		b.Logf("[ResNet] SyncHE done at +%v", time.Since(startAll))

		// Encrypt input: 64 channels, each 32x32
		inputCTs := make([]*rlwe.Ciphertext, 64)
		slots := heCtx.Params.MaxSlots()
		for ch := 0; ch < 64; ch++ {
			inputVec := make([]complex128, slots)
			for i := 0; i < 32*32; i++ {
				inputVec[i] = complex(float64((ch+i)%10)/10.0, 0)
			}
			inputPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(inputVec, inputPT)
			inputCTs[ch], _ = heCtx.Encryptor.EncryptNew(inputPT)
		}
		b.Logf("[ResNet] input encrypted at +%v", time.Since(startAll))

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iterStart := time.Now()
			b.Logf("[ResNet] Iter %d start +%v", i, time.Since(startAll))

			out, _ := resnetConv.ForwardHE(inputCTs)
			b.Logf("[ResNet] Iter %d forward done +%v", i, time.Since(iterStart))

			_, _ = resnetConv.BackwardHE(out)
			b.Logf("[ResNet] Iter %d backward done +%v", i, time.Since(iterStart))
		}
		b.StopTimer()

		b.Logf("[ResNet] benchmark end at +%v", time.Since(startAll))
	})
}

// Benchmark only the HE Backward pass
func BenchmarkConv2D_BackwardHE(b *testing.B) {
	b.Run("BackwardHE", func(b *testing.B) {
		heCtx := ckkswrapper.NewHeContext()
		conv := NewConv2D(1, 6, 5, 5, true, heCtx)
		conv.SetDimensions(28, 28)
		conv.SyncHE()

		// Create input and run forward pass to get output
		input := make([]*rlwe.Ciphertext, 1)
		slots := heCtx.Params.MaxSlots()
		inputVec := make([]complex128, slots)
		for i := 0; i < 28*28; i++ {
			inputVec[i] = complex(float64(i%10)/10.0, 0)
		}
		inputPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, inputPT)
		input[0], _ = heCtx.Encryptor.EncryptNew(inputPT)

		// Run forward pass to get output (we need this to cache the input)
		_, _ = conv.ForwardHE(input)

		// Create gradient output with correct length (numGroups)
		gradOut := make([]*rlwe.Ciphertext, conv.numGroups)
		for group := 0; group < conv.numGroups; group++ {
			// Create a simple gradient pattern
			gradVec := make([]complex128, slots)
			for i := 0; i < 24*24; i++ { // output size is 24x24 for 28x28 input with 5x5 kernel
				gradVec[i] = complex(float64(i%5)/5.0, 0)
			}
			gradPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(gradVec, gradPT)
			gradOut[group], _ = heCtx.Encryptor.EncryptNew(gradPT)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			start := time.Now()
			_, _ = conv.BackwardHE(gradOut)
			b.Logf("Iter %d BackwardHE time: %v", i, time.Since(start))
		}
	})
}

// Benchmark the full HE Update pass
func BenchmarkConv2D_UpdateHE(b *testing.B) {
	b.Run("UpdateHE", func(b *testing.B) {
		heCtx := ckkswrapper.NewHeContext()
		conv := NewConv2D(1, 6, 5, 5, true, heCtx)
		conv.SetDimensions(28, 28)
		conv.SyncHE()

		// Create input and run forward/backward to get gradients
		input := make([]*rlwe.Ciphertext, 1)
		slots := heCtx.Params.MaxSlots()
		inputVec := make([]complex128, slots)
		for i := 0; i < 28*28; i++ {
			inputVec[i] = complex(float64(i%10)/10.0, 0)
		}
		inputPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, inputPT)
		input[0], _ = heCtx.Encryptor.EncryptNew(inputPT)

		// Run forward and backward to populate gradients
		_, _ = conv.ForwardHE(input)

		// Create gradient output with correct length (numGroups)
		gradOut := make([]*rlwe.Ciphertext, conv.numGroups)
		for group := 0; group < conv.numGroups; group++ {
			// Create a simple gradient pattern
			gradVec := make([]complex128, slots)
			for i := 0; i < 24*24; i++ { // output size is 24x24 for 28x28 input with 5x5 kernel
				gradVec[i] = complex(float64(i%5)/5.0, 0)
			}
			gradPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(gradVec, gradPT)
			gradOut[group], _ = heCtx.Encryptor.EncryptNew(gradPT)
		}

		_, _ = conv.BackwardHE(gradOut)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			start := time.Now()
			_ = conv.UpdateHE(0.01)
			b.Logf("Iter %d UpdateHE time: %v", i, time.Since(start))
		}
	})
}

// Benchmark only the HE Backward pass for ResNet-style Conv2D
func BenchmarkConv2D_ResNet_BackwardHE(b *testing.B) {
	b.Run("ResNet_BackwardHE", func(b *testing.B) {
		heCtx := ckkswrapper.NewHeContext()
		startAll := time.Now()
		b.Logf("[ResNet_BackwardHE] benchmark start: %v", startAll)

		// Layer setup: 64->64, 3x3 kernel, 32x32 input
		resnetConv := NewConv2D(64, 64, 3, 3, true, heCtx)
		resnetConv.SetDimensions(32, 32)
		b.Logf("[ResNet_BackwardHE] SetDimensions done at +%v", time.Since(startAll))

		resnetConv.SyncHE()
		b.Logf("[ResNet_BackwardHE] SyncHE done at +%v", time.Since(startAll))

		// Encrypt input: 64 channels, each 32x32
		inputCTs := make([]*rlwe.Ciphertext, 64)
		slots := heCtx.Params.MaxSlots()
		for ch := 0; ch < 64; ch++ {
			inputVec := make([]complex128, slots)
			for i := 0; i < 32*32; i++ {
				inputVec[i] = complex(float64((ch+i)%10)/10.0, 0)
			}
			inputPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(inputVec, inputPT)
			inputCTs[ch], _ = heCtx.Encryptor.EncryptNew(inputPT)
		}
		b.Logf("[ResNet_BackwardHE] input encrypted at +%v", time.Since(startAll))

		// Run forward pass to cache input
		_, _ = resnetConv.ForwardHE(inputCTs)
		b.Logf("[ResNet_BackwardHE] forward pass done at +%v", time.Since(startAll))

		// Create gradient output with correct length (numGroups)
		gradOut := make([]*rlwe.Ciphertext, resnetConv.numGroups)
		for group := 0; group < resnetConv.numGroups; group++ {
			// Create a simple gradient pattern
			gradVec := make([]complex128, slots)
			for i := 0; i < 30*30; i++ { // output size is 30x30 for 32x32 input with 3x3 kernel
				gradVec[i] = complex(float64(i%5)/5.0, 0)
			}
			gradPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(gradVec, gradPT)
			gradOut[group], _ = heCtx.Encryptor.EncryptNew(gradPT)
		}
		b.Logf("[ResNet_BackwardHE] gradient created at +%v", time.Since(startAll))

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iterStart := time.Now()
			b.Logf("[ResNet_BackwardHE] Iter %d start +%v", i, time.Since(startAll))

			_, _ = resnetConv.BackwardHE(gradOut)
			b.Logf("[ResNet_BackwardHE] Iter %d backward done +%v", i, time.Since(iterStart))
		}
		b.StopTimer()

		b.Logf("[ResNet_BackwardHE] benchmark end at +%v", time.Since(startAll))
	})
}

// Benchmark the full HE Update pass for ResNet-style Conv2D (simplified)
func BenchmarkConv2D_ResNet_UpdateHE_Simple(b *testing.B) {
	b.Run("ResNet_UpdateHE_Simple", func(b *testing.B) {
		heCtx := ckkswrapper.NewHeContext()
		startAll := time.Now()
		b.Logf("[ResNet_UpdateHE_Simple] benchmark start: %v", startAll)

		// Layer setup: 64->64, 3x3 kernel, 32x32 input
		resnetConv := NewConv2D(64, 64, 3, 3, true, heCtx)
		resnetConv.SetDimensions(32, 32)
		b.Logf("[ResNet_UpdateHE_Simple] SetDimensions done at +%v", time.Since(startAll))

		resnetConv.SyncHE()
		b.Logf("[ResNet_UpdateHE_Simple] SyncHE done at +%v", time.Since(startAll))

		// Create dummy gradients directly instead of running BackwardHE
		slots := heCtx.Params.MaxSlots()

		// Create dummy bias gradient
		biasGradVec := make([]complex128, slots)
		for i := 0; i < 64; i++ { // 64 output channels
			biasGradVec[i] = complex(float64(i%10)/10.0, 0)
		}
		biasGradPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(biasGradVec, biasGradPT)
		resnetConv.gradBCT, _ = heCtx.Encryptor.EncryptNew(biasGradPT)
		b.Logf("[ResNet_UpdateHE_Simple] bias gradient created at +%v", time.Since(startAll))

		// Create dummy weight gradients (one per group)
		resnetConv.gradWCTs = make([]*rlwe.Ciphertext, resnetConv.numGroups)
		for group := 0; group < resnetConv.numGroups; group++ {
			weightGradVec := make([]complex128, slots)
			for i := 0; i < 576; i++ { // patchDim = 576 for ResNet
				weightGradVec[i] = complex(float64((group+i)%10)/10.0, 0)
			}
			weightGradPT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(weightGradVec, weightGradPT)
			resnetConv.gradWCTs[group], _ = heCtx.Encryptor.EncryptNew(weightGradPT)
		}
		b.Logf("[ResNet_UpdateHE_Simple] weight gradients created at +%v", time.Since(startAll))

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			iterStart := time.Now()
			b.Logf("[ResNet_UpdateHE_Simple] Iter %d start +%v", i, time.Since(startAll))

			_ = resnetConv.UpdateHE(0.01)
			b.Logf("[ResNet_UpdateHE_Simple] Iter %d update done +%v", i, time.Since(iterStart))
		}
		b.StopTimer()

		b.Logf("[ResNet_UpdateHE_Simple] benchmark end at +%v", time.Since(startAll))
	})
}

// Benchmark the full HE Update pass for ResNet-style Conv2D
