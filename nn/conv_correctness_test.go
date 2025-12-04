package nn

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/tensor"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ConvLayerDivergence stores divergence metrics for a convolutional layer
type ConvLayerDivergence struct {
	LayerName     string
	LayerType     string // "Conv2D", "Conv1D", "Activation"
	MaxAbsError   float64
	MeanAbsError  float64
	RMSError      float64
	MaxRelError   float64
	NumDimensions int
}

// ConvCorrectnessReport stores the full report for conv tests
type ConvCorrectnessReport struct {
	ModelName         string
	NumLayers         int
	ForwardDivergence []ConvLayerDivergence
	TotalForwardError float64
}

func (r *ConvCorrectnessReport) PrintReport() string {
	var sb strings.Builder
	sb.WriteString("\n")
	sb.WriteString(strings.Repeat("═", 100) + "\n")
	sb.WriteString(fmt.Sprintf("CONV LAYER HE vs PLAINTEXT CORRECTNESS REPORT: %s\n", r.ModelName))
	sb.WriteString(strings.Repeat("═", 100) + "\n")
	sb.WriteString(fmt.Sprintf("Number of Layers: %d\n", r.NumLayers))
	sb.WriteString(strings.Repeat("-", 100) + "\n")
	sb.WriteString(fmt.Sprintf("%-5s | %-30s | %-12s | %-12s | %-12s | %-12s | %-6s\n",
		"Index", "Layer Name", "Max Abs Err", "Mean Abs Err", "RMS Err", "Max Rel Err", "Dims"))
	sb.WriteString(strings.Repeat("-", 100) + "\n")

	for i, div := range r.ForwardDivergence {
		sb.WriteString(fmt.Sprintf("%-5d | %-30s | %.4e | %.4e | %.4e | %.4e | %-6d\n",
			i, div.LayerName, div.MaxAbsError, div.MeanAbsError, div.RMSError, div.MaxRelError, div.NumDimensions))
	}
	sb.WriteString(strings.Repeat("-", 100) + "\n")
	sb.WriteString(fmt.Sprintf("Total Forward RMS Error: %.6e\n", r.TotalForwardError))
	sb.WriteString(strings.Repeat("═", 100) + "\n")
	return sb.String()
}

// computeConvDivergence computes divergence metrics between HE and plaintext outputs
func computeConvDivergence(heVals, plainVals []float64, layerName, layerType string) ConvLayerDivergence {
	n := len(heVals)
	if len(plainVals) < n {
		n = len(plainVals)
	}

	var maxAbs, sumAbs, sumSq, maxRel float64
	for i := 0; i < n; i++ {
		diff := math.Abs(heVals[i] - plainVals[i])
		if diff > maxAbs {
			maxAbs = diff
		}
		sumAbs += diff
		sumSq += diff * diff

		// Relative error
		if math.Abs(plainVals[i]) > 1e-10 {
			rel := diff / math.Abs(plainVals[i])
			if rel > maxRel {
				maxRel = rel
			}
		}
	}

	return ConvLayerDivergence{
		LayerName:     layerName,
		LayerType:     layerType,
		MaxAbsError:   maxAbs,
		MeanAbsError:  sumAbs / float64(n),
		RMSError:      math.Sqrt(sumSq / float64(n)),
		MaxRelError:   maxRel,
		NumDimensions: n,
	}
}

// decryptConvOutput decrypts ciphertext slices and returns float64 values
// The output is packed with inW stride: position (oy, ox) is at slot oy*inW + ox
func decryptConvOutput(heCtx *ckkswrapper.HeContext, cts []*rlwe.Ciphertext, outH, outW, inW int) []float64 {
	var result []float64
	for _, ct := range cts {
		pt := heCtx.Decryptor.DecryptNew(ct)
		decoded := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(pt, decoded)

		// Extract output values from packed positions
		for oy := 0; oy < outH; oy++ {
			for ox := 0; ox < outW; ox++ {
				pos := oy*inW + ox
				result = append(result, real(decoded[pos]))
			}
		}
	}
	return result
}

// decryptConvOutputSimple decrypts ciphertext slices assuming consecutive packing
// Used for Conv1D (outH=1) where positions are consecutive, or for simple extraction
func decryptConvOutputSimple(heCtx *ckkswrapper.HeContext, cts []*rlwe.Ciphertext, numVals int) []float64 {
	var result []float64
	for _, ct := range cts {
		pt := heCtx.Decryptor.DecryptNew(ct)
		decoded := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(pt, decoded)

		// Extract consecutive values from positions 0 to numVals-1
		for i := 0; i < numVals; i++ {
			result = append(result, real(decoded[i]))
		}
	}
	return result
}

// ============================================================================
// CONV2D TESTS
// ============================================================================// TestConv2DCorrectness tests a single Conv2D layer HE vs plaintext
func TestConv2DCorrectness(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// Conv2D: 1 input channel -> 1 output channel, 2x2 kernel
	inChan, outChan := 1, 1
	kh, kw := 2, 2
	inH, inW := 3, 3

	// Create HE and plaintext conv layers
	heConv := layers.NewConv2D(inChan, outChan, kh, kw, true, heCtx)
	plainConv := layers.NewConv2D(inChan, outChan, kh, kw, false, nil)

	// Initialize with same weights = all 1.0
	for oc := 0; oc < outChan; oc++ {
		for ic := 0; ic < inChan; ic++ {
			for i := 0; i < kh; i++ {
				for j := 0; j < kw; j++ {
					heConv.W.Set(1.0, oc, ic, i, j)
					plainConv.W.Set(1.0, oc, ic, i, j)
				}
			}
		}
		heConv.B.Set(0.0, oc)
		plainConv.B.Set(0.0, oc)
	}

	// Set dimensions and sync HE
	_ = heConv.SetDimensions(inH, inW)
	_ = heConv.SyncHE()

	// Create simple input: 1,2,3,4,5,6,7,8,9
	inputTensor := tensor.New(inChan, inH, inW)
	for i := range inputTensor.Data {
		inputTensor.Data[i] = float64(i + 1)
	}

	// Plaintext forward - expected: [12, 16, 24, 28]
	plainOutput, _ := plainConv.ForwardPlain(inputTensor)
	t.Logf("Plaintext output: %v", plainOutput.Data)

	// Encrypt and run HE forward
	slots := heCtx.Params.MaxSlots()
	inputCTs := make([]*rlwe.Ciphertext, inChan)
	for c := 0; c < inChan; c++ {
		inputVec := make([]complex128, slots)
		for i := 0; i < inH*inW; i++ {
			inputVec[i] = complex(inputTensor.Data[c*inH*inW+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		inputCTs[c] = ct
	}

	heOutputCTs, err := heConv.ForwardHE(inputCTs)
	if err != nil {
		t.Fatalf("ForwardHE failed: %v", err)
	}

	// Decrypt - output is packed with inW stride (positions 0, 1 for row 0; positions 3, 4 for row 1)
	pt := heCtx.Decryptor.DecryptNew(heOutputCTs[0])
	decoded := make([]complex128, slots)
	heCtx.Encoder.Decode(pt, decoded)

	// Extract output values - output is at positions oy*inW + ox
	outH := inH - kh + 1
	outW := inW - kw + 1
	heOutput := make([]float64, outH*outW)
	for oy := 0; oy < outH; oy++ {
		for ox := 0; ox < outW; ox++ {
			pos := oy*inW + ox
			heOutput[oy*outW+ox] = real(decoded[pos])
		}
	}

	t.Logf("HE output (extracted): %v", heOutput)

	// Compare with plaintext
	for i := 0; i < outH*outW; i++ {
		diff := math.Abs(heOutput[i] - plainOutput.Data[i])
		t.Logf("  [%d] HE=%.6f, Plain=%.6f, Diff=%.6e", i, heOutput[i], plainOutput.Data[i], diff)
		if diff > 1e-5 {
			t.Errorf("Position %d: expected %.6f, got %.6f (diff=%.6e)", i, plainOutput.Data[i], heOutput[i], diff)
		}
	}
}

// TestConv2DMultiChannel tests Conv2D with multiple input/output channels
func TestConv2DMultiChannel(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// Conv2D: 3 input channels -> 4 output channels, 3x3 kernel
	inChan, outChan := 3, 4
	kh, kw := 3, 3
	inH, inW := 8, 8

	heConv := layers.NewConv2D(inChan, outChan, kh, kw, true, heCtx)
	plainConv := layers.NewConv2D(inChan, outChan, kh, kw, false, nil)

	// Initialize weights
	scale := math.Sqrt(2.0 / float64(inChan*kh*kw))
	for oc := 0; oc < outChan; oc++ {
		for ic := 0; ic < inChan; ic++ {
			for i := 0; i < kh; i++ {
				for j := 0; j < kw; j++ {
					w := (rand.Float64() - 0.5) * 2 * scale
					heConv.W.Set(w, oc, ic, i, j)
					plainConv.W.Set(w, oc, ic, i, j)
				}
			}
		}
		b := (rand.Float64() - 0.5) * 0.1
		heConv.B.Set(b, oc)
		plainConv.B.Set(b, oc)
	}

	heConv.SetDimensions(inH, inW)
	heConv.SyncHE()

	// Create input
	inputTensor := tensor.New(inChan, inH, inW)
	for i := range inputTensor.Data {
		inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
	}

	// Forward passes
	plainOutput, _ := plainConv.ForwardPlain(inputTensor)

	slots := heCtx.Params.MaxSlots()
	inputCTs := make([]*rlwe.Ciphertext, inChan)
	for c := 0; c < inChan; c++ {
		inputVec := make([]complex128, slots)
		for i := 0; i < inH*inW; i++ {
			inputVec[i] = complex(inputTensor.Data[c*inH*inW+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		inputCTs[c] = ct
	}

	heOutputCTs, err := heConv.ForwardHE(inputCTs)
	if err != nil {
		t.Fatalf("HE forward failed: %v", err)
	}

	outH := inH - kh + 1
	outW := inW - kw + 1
	heOutputVals := decryptConvOutput(heCtx, heOutputCTs, outH, outW, inW)

	div := computeConvDivergence(heOutputVals, plainOutput.Data[:len(heOutputVals)],
		fmt.Sprintf("Conv2D(%d->%d, %dx%d)", inChan, outChan, kh, kw), "Conv2D")

	t.Logf("\n=== Conv2D Multi-Channel Test ===")
	t.Logf("Input: %d channels, %dx%d spatial", inChan, inH, inW)
	t.Logf("Output: %d channels, %dx%d spatial", outChan, outH, outW)
	t.Logf("RMS Error: %.6e", div.RMSError)
}

// ============================================================================
// CONV1D TESTS
// ============================================================================

// TestConv1DCorrectness tests a single Conv1D layer HE vs plaintext
func TestConv1DCorrectness(t *testing.T) {

	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// Conv1D: 1 input channel -> 2 output channels, kernel size 3
	inChan, outChan := 1, 2
	kernelSize := 3
	seqLen := 16

	heConv := layers.NewConv1D(inChan, outChan, kernelSize, true, heCtx)
	plainConv := layers.NewConv1D(inChan, outChan, kernelSize, false, nil)

	// Initialize weights
	scale := math.Sqrt(2.0 / float64(inChan*kernelSize))
	for oc := 0; oc < outChan; oc++ {
		for ic := 0; ic < inChan; ic++ {
			for k := 0; k < kernelSize; k++ {
				w := (rand.Float64() - 0.5) * 2 * scale
				heConv.W.Set(w, oc, ic, 0, k)
				plainConv.W.Set(w, oc, ic, 0, k)
			}
		}
		b := (rand.Float64() - 0.5) * 0.1
		heConv.B.Set(b, oc)
		plainConv.B.Set(b, oc)
	}

	// Set dimensions (1D conv: height=1, width=seqLen)
	heConv.SetDimensions(1, seqLen)
	heConv.SyncHE()

	// Create input
	inputTensor := tensor.New(inChan, 1, seqLen)
	for i := range inputTensor.Data {
		inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
	}

	// Forward passes
	plainOutput, _ := plainConv.ForwardPlain(inputTensor)

	slots := heCtx.Params.MaxSlots()
	inputCTs := make([]*rlwe.Ciphertext, inChan)
	for c := 0; c < inChan; c++ {
		inputVec := make([]complex128, slots)
		for i := 0; i < seqLen; i++ {
			inputVec[i] = complex(inputTensor.Data[c*seqLen+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		inputCTs[c] = ct
	}

	heOutputCTs, err := heConv.ForwardHE(inputCTs)
	if err != nil {
		t.Fatalf("HE forward failed: %v", err)
	}

	outLen := seqLen - kernelSize + 1
	// For Conv1D: outH=1, outW=outLen, inW=seqLen. Positions are at 0*seqLen+ox = ox (consecutive)
	heOutputVals := decryptConvOutput(heCtx, heOutputCTs, 1, outLen, seqLen)

	div := computeConvDivergence(heOutputVals, plainOutput.Data[:len(heOutputVals)],
		fmt.Sprintf("Conv1D(%d->%d, k=%d)", inChan, outChan, kernelSize), "Conv1D")

	t.Logf("\n=== Conv1D Correctness Test ===")
	t.Logf("Input: %d channels, length %d", inChan, seqLen)
	t.Logf("Output: %d channels, length %d", outChan, outLen)
	t.Logf("RMS Error: %.6e", div.RMSError)
}

// ============================================================================
// CONV + ACTIVATION PIPELINE TESTS
// ============================================================================

// TestConv2DWithActivation tests Conv2D -> ReLU3 pipeline
func TestConv2DWithActivation(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// Conv2D: 1 -> 2 channels, 3x3 kernel
	inChan, outChan := 1, 2
	kh, kw := 3, 3
	inH, inW := 8, 8

	heConv := layers.NewConv2D(inChan, outChan, kh, kw, true, heCtx)
	plainConv := layers.NewConv2D(inChan, outChan, kh, kw, false, nil)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Initialize weights
	scale := math.Sqrt(2.0 / float64(inChan*kh*kw))
	for oc := 0; oc < outChan; oc++ {
		for ic := 0; ic < inChan; ic++ {
			for i := 0; i < kh; i++ {
				for j := 0; j < kw; j++ {
					w := (rand.Float64() - 0.5) * 2 * scale
					heConv.W.Set(w, oc, ic, i, j)
					plainConv.W.Set(w, oc, ic, i, j)
				}
			}
		}
		b := (rand.Float64() - 0.5) * 0.1
		heConv.B.Set(b, oc)
		plainConv.B.Set(b, oc)
	}

	heConv.SetDimensions(inH, inW)
	heConv.SyncHE()

	// Create input
	inputTensor := tensor.New(inChan, inH, inW)
	for i := range inputTensor.Data {
		inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
	}

	// Plaintext forward: Conv -> True ReLU
	plainConvOut, _ := plainConv.ForwardPlain(inputTensor)

	// HE forward: Conv
	slots := heCtx.Params.MaxSlots()
	inputCTs := make([]*rlwe.Ciphertext, inChan)
	for c := 0; c < inChan; c++ {
		inputVec := make([]complex128, slots)
		for i := 0; i < inH*inW; i++ {
			inputVec[i] = complex(inputTensor.Data[c*inH*inW+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		inputCTs[c] = ct
	}

	heConvOut, _ := heConv.ForwardHE(inputCTs)

	outH := inH - kh + 1
	outW := inW - kw + 1

	report := &ConvCorrectnessReport{
		ModelName: fmt.Sprintf("Conv2D(%d->%d) + ReLU3", inChan, outChan),
		NumLayers: 2,
	}

	// Measure Conv2D divergence
	heConvVals := decryptConvOutput(heCtx, heConvOut, outH, outW, inW)
	convDiv := computeConvDivergence(heConvVals, plainConvOut.Data[:len(heConvVals)],
		fmt.Sprintf("Conv2D(%d->%d, %dx%d)", inChan, outChan, kh, kw), "Conv2D")
	report.ForwardDivergence = append(report.ForwardDivergence, convDiv)

	// Apply activation to each channel
	heActOut := make([]*rlwe.Ciphertext, len(heConvOut))
	plainActOut := make([]float64, len(plainConvOut.Data))

	for c := 0; c < len(heConvOut); c++ {
		actOut, err := heActivation.ForwardCipher(heConvOut[c])
		if err != nil {
			t.Fatalf("HE activation failed: %v", err)
		}
		heActOut[c] = actOut
	}

	// Apply ReLU3 polynomial to plaintext (same polynomial as HE)
	// ReLU3 coefficients: [0.3183099, 0.5, 0.2122066, 0]
	for i, x := range plainConvOut.Data {
		plainActOut[i] = 0.3183099 + 0.5*x + 0.2122066*x*x // degree 3 with c3=0
	}

	// Measure Activation divergence (HE ReLU3 vs Plaintext ReLU3)
	heActVals := decryptConvOutput(heCtx, heActOut, outH, outW, inW)
	actDiv := computeConvDivergence(heActVals, plainActOut[:len(heActVals)],
		"Activation (HE vs Plain ReLU3)", "Activation")
	report.ForwardDivergence = append(report.ForwardDivergence, actDiv)

	// Calculate total RMS
	var totalRMS float64
	for _, div := range report.ForwardDivergence {
		totalRMS += div.RMSError * div.RMSError
	}
	report.TotalForwardError = math.Sqrt(totalRMS)

	t.Log(report.PrintReport())
}

// ============================================================================
// COMPREHENSIVE CNN ARCHITECTURE TESTS
// ============================================================================

// TestLeNetConvLayers tests LeNet-style conv layers
func TestLeNetConvLayers(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	t.Log("\n=== LeNet Conv Layer Test ===")
	t.Log("Architecture: Conv(1->6, 5x5) -> ReLU3")

	// LeNet first conv: 1 input channel, 6 output channels, 5x5 kernel
	inChan, outChan := 1, 6
	kh, kw := 5, 5
	inH, inW := 16, 16 // Smaller than MNIST's 28x28 for faster test

	heConv := layers.NewConv2D(inChan, outChan, kh, kw, true, heCtx)
	plainConv := layers.NewConv2D(inChan, outChan, kh, kw, false, nil)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Initialize
	scale := math.Sqrt(2.0 / float64(inChan*kh*kw))
	for oc := 0; oc < outChan; oc++ {
		for ic := 0; ic < inChan; ic++ {
			for i := 0; i < kh; i++ {
				for j := 0; j < kw; j++ {
					w := (rand.Float64() - 0.5) * 2 * scale
					heConv.W.Set(w, oc, ic, i, j)
					plainConv.W.Set(w, oc, ic, i, j)
				}
			}
		}
		heConv.B.Set((rand.Float64()-0.5)*0.1, oc)
		plainConv.B.Set(heConv.B.At(oc), oc)
	}

	heConv.SetDimensions(inH, inW)
	heConv.SyncHE()

	// Input
	inputTensor := tensor.New(inChan, inH, inW)
	for i := range inputTensor.Data {
		inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
	}

	// Forward passes
	plainConvOut, _ := plainConv.ForwardPlain(inputTensor)

	slots := heCtx.Params.MaxSlots()
	inputCTs := make([]*rlwe.Ciphertext, inChan)
	for c := 0; c < inChan; c++ {
		inputVec := make([]complex128, slots)
		for i := 0; i < inH*inW; i++ {
			inputVec[i] = complex(inputTensor.Data[c*inH*inW+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		inputCTs[c] = ct
	}

	heConvOut, err := heConv.ForwardHE(inputCTs)
	if err != nil {
		t.Fatalf("HE Conv forward failed: %v", err)
	}

	outH := inH - kh + 1
	outW := inW - kw + 1

	// Conv divergence
	heConvVals := decryptConvOutput(heCtx, heConvOut, outH, outW, inW)
	convDiv := computeConvDivergence(heConvVals, plainConvOut.Data[:len(heConvVals)],
		"Conv2D(1->6, 5x5)", "Conv2D")
	t.Logf("Conv2D RMS Error: %.6e", convDiv.RMSError)

	// Apply activation
	heActOut := make([]*rlwe.Ciphertext, len(heConvOut))
	for c := 0; c < len(heConvOut); c++ {
		actOut, _ := heActivation.ForwardCipher(heConvOut[c])
		heActOut[c] = actOut
	}

	// Apply ReLU3 polynomial to plaintext (same as HE)
	plainActOut := make([]float64, len(plainConvOut.Data))
	for i, x := range plainConvOut.Data {
		plainActOut[i] = 0.3183099 + 0.5*x + 0.2122066*x*x // ReLU3 polynomial
	}

	heActVals := decryptConvOutput(heCtx, heActOut, outH, outW, inW)
	actDiv := computeConvDivergence(heActVals, plainActOut[:len(heActVals)],
		"Activation (HE vs Plain ReLU3)", "Activation")
	t.Logf("Activation RMS Error: %.6e (HE vs plaintext ReLU3)", actDiv.RMSError)

	t.Logf("Total RMS Error: %.6e", math.Sqrt(convDiv.RMSError*convDiv.RMSError+actDiv.RMSError*actDiv.RMSError))
}

// TestAllConvArchitectures runs all conv architectures and outputs CSV
func TestAllConvArchitectures(t *testing.T) {
	type ConvArch struct {
		name     string
		inChan   int
		outChan  int
		kh, kw   int
		inH, inW int
	}

	architectures := []ConvArch{
		{"LeNet_Conv1", 1, 6, 5, 5, 16, 16},
		{"LeNet_Conv2", 6, 16, 5, 5, 8, 8},
		{"Small_Conv", 1, 2, 3, 3, 8, 8},
		{"Audio_Conv1D", 1, 4, 1, 3, 1, 32}, // 1D conv as 1xL (kh=1, kw=3)
	}

	t.Log("\n" + strings.Repeat("=", 100))
	t.Log("COMPREHENSIVE CONV LAYER DIVERGENCE ANALYSIS")
	t.Log(strings.Repeat("=", 100))

	t.Log("\n" + strings.Repeat("-", 100))
	t.Logf("%-15s | %-20s | %-15s | %-15s | %-15s", "Model", "Config", "Conv RMS", "Act RMS", "Total RMS")
	t.Log(strings.Repeat("-", 100))

	for _, arch := range architectures {
		rand.Seed(42)
		heCtx := ckkswrapper.NewHeContext()

		heConv := layers.NewConv2D(arch.inChan, arch.outChan, arch.kh, arch.kw, true, heCtx)
		plainConv := layers.NewConv2D(arch.inChan, arch.outChan, arch.kh, arch.kw, false, nil)
		heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

		// Initialize
		scale := math.Sqrt(2.0 / float64(arch.inChan*arch.kh*arch.kw))
		for oc := 0; oc < arch.outChan; oc++ {
			for ic := 0; ic < arch.inChan; ic++ {
				for i := 0; i < arch.kh; i++ {
					for j := 0; j < arch.kw; j++ {
						w := (rand.Float64() - 0.5) * 2 * scale
						heConv.W.Set(w, oc, ic, i, j)
						plainConv.W.Set(w, oc, ic, i, j)
					}
				}
			}
			heConv.B.Set((rand.Float64()-0.5)*0.1, oc)
			plainConv.B.Set(heConv.B.At(oc), oc)
		}

		heConv.SetDimensions(arch.inH, arch.inW)
		heConv.SyncHE()

		// Input
		inputTensor := tensor.New(arch.inChan, arch.inH, arch.inW)
		for i := range inputTensor.Data {
			inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
		}

		// Forward
		plainConvOut, _ := plainConv.ForwardPlain(inputTensor)

		slots := heCtx.Params.MaxSlots()
		inputCTs := make([]*rlwe.Ciphertext, arch.inChan)
		for c := 0; c < arch.inChan; c++ {
			inputVec := make([]complex128, slots)
			for i := 0; i < arch.inH*arch.inW; i++ {
				inputVec[i] = complex(inputTensor.Data[c*arch.inH*arch.inW+i], 0)
			}
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			heCtx.Encoder.Encode(inputVec, pt)
			ct, _ := heCtx.Encryptor.EncryptNew(pt)
			inputCTs[c] = ct
		}

		heConvOut, err := heConv.ForwardHE(inputCTs)
		if err != nil {
			t.Logf("%-15s | FAILED: %v", arch.name, err)
			continue
		}

		outH := arch.inH - arch.kh + 1
		outW := arch.inW - arch.kw + 1

		// Conv divergence
		heConvVals := decryptConvOutput(heCtx, heConvOut, outH, outW, arch.inW)
		convDiv := computeConvDivergence(heConvVals, plainConvOut.Data[:len(heConvVals)],
			arch.name, "Conv2D")

		// Activation
		heActOut := make([]*rlwe.Ciphertext, len(heConvOut))
		for c := 0; c < len(heConvOut); c++ {
			actOut, _ := heActivation.ForwardCipher(heConvOut[c])
			heActOut[c] = actOut
		}

		// Apply ReLU3 polynomial to plaintext (same as HE)
		plainActOut := make([]float64, len(plainConvOut.Data))
		for i, x := range plainConvOut.Data {
			plainActOut[i] = 0.3183099 + 0.5*x + 0.2122066*x*x // ReLU3 polynomial
		}

		heActVals := decryptConvOutput(heCtx, heActOut, outH, outW, arch.inW)
		actDiv := computeConvDivergence(heActVals, plainActOut[:len(heActVals)],
			"Activation (HE vs Plain)", "Activation")

		totalRMS := math.Sqrt(convDiv.RMSError*convDiv.RMSError + actDiv.RMSError*actDiv.RMSError)

		config := fmt.Sprintf("%d->%d, %dx%d", arch.inChan, arch.outChan, arch.kh, arch.kw)
		t.Logf("%-15s | %-20s | %.6e    | %.6e    | %.6e",
			arch.name, config, convDiv.RMSError, actDiv.RMSError, totalRMS)
	}

	t.Log(strings.Repeat("-", 100))
}
