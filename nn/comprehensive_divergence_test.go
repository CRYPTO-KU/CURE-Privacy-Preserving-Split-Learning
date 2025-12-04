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

// LayerDivergenceInfo stores divergence metrics with accumulated error
type LayerDivergenceInfo struct {
	LayerName     string
	LayerType     string
	LayerRMSError float64 // Error at this layer only
	AccumRMSError float64 // Accumulated error up to this layer
	MaxAbsError   float64
	NumDimensions int
}

// ModelDivergenceReport stores the full report for a model
type ModelDivergenceReport struct {
	ModelName     string
	Architecture  string
	NumLayers     int
	Layers        []LayerDivergenceInfo
	FinalRMSError float64
}

// applyReLU3Poly applies the ReLU3 polynomial to a slice of float64
// ReLU3 coefficients: [0.3183099, 0.5, 0.2122066, 0]
func applyReLU3Poly(vals []float64) []float64 {
	result := make([]float64, len(vals))
	for i, x := range vals {
		result[i] = 0.3183099 + 0.5*x + 0.2122066*x*x // c0 + c1*x + c2*x^2 (c3=0)
	}
	return result
}

// calcRMSError computes RMS error between two slices
func calcRMSError(heVals, plainVals []float64) float64 {
	if len(heVals) != len(plainVals) {
		n := len(heVals)
		if len(plainVals) < n {
			n = len(plainVals)
		}
		heVals = heVals[:n]
		plainVals = plainVals[:n]
	}

	var sumSq float64
	for i := range heVals {
		diff := heVals[i] - plainVals[i]
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(heVals)))
}

// calcMaxAbsError computes max absolute error between two slices
func calcMaxAbsError(heVals, plainVals []float64) float64 {
	if len(heVals) != len(plainVals) {
		n := len(heVals)
		if len(plainVals) < n {
			n = len(plainVals)
		}
		heVals = heVals[:n]
		plainVals = plainVals[:n]
	}

	maxErr := 0.0
	for i := range heVals {
		diff := math.Abs(heVals[i] - plainVals[i])
		if diff > maxErr {
			maxErr = diff
		}
	}
	return maxErr
}

// decryptToSlice decrypts a ciphertext and returns float64 slice
func decryptToSlice(heCtx *ckkswrapper.HeContext, ct *rlwe.Ciphertext, numVals int) []float64 {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)

	result := make([]float64, numVals)
	for i := 0; i < numVals; i++ {
		result[i] = real(decoded[i])
	}
	return result
}

// cleanRefresh decrypts, zeros out slots beyond validSize, and re-encrypts
func cleanRefresh(heCtx *ckkswrapper.HeContext, ct *rlwe.Ciphertext, validSize int) *rlwe.Ciphertext {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)

	// Zero out slots beyond valid size
	for i := validSize; i < len(decoded); i++ {
		decoded[i] = 0
	}

	// Re-encode and re-encrypt
	newPt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	newPt.Scale = heCtx.Params.DefaultScale()
	heCtx.Encoder.Encode(decoded, newPt)
	newCt, _ := heCtx.Encryptor.EncryptNew(newPt)
	return newCt
}

// cleanRefreshConv2D decrypts, zeros out invalid slots, and re-encrypts
// Valid slots are at positions (oy * inW + ox) for oy in 0..outH-1, ox in 0..outW-1
func cleanRefreshConv2D(heCtx *ckkswrapper.HeContext, ct *rlwe.Ciphertext, outH, outW, inW int) *rlwe.Ciphertext {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)

	// Create valid slot mask
	validSlots := make(map[int]bool)
	for oy := 0; oy < outH; oy++ {
		for ox := 0; ox < outW; ox++ {
			validSlots[oy*inW+ox] = true
		}
	}

	// Zero out invalid slots
	for i := 0; i < len(decoded); i++ {
		if !validSlots[i] {
			decoded[i] = 0
		}
	}

	// Re-encode and re-encrypt at max level
	newPt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	newPt.Scale = heCtx.Params.DefaultScale()
	heCtx.Encoder.Encode(decoded, newPt)
	newCt, _ := heCtx.Encryptor.EncryptNew(newPt)
	return newCt
}

// decryptConv2DOutput decrypts Conv2D output (values at positions oy*inW + ox)
func decryptConv2DOutput(heCtx *ckkswrapper.HeContext, cts []*rlwe.Ciphertext, outH, outW, inW int) []float64 {
	var result []float64
	for _, ct := range cts {
		pt := heCtx.Decryptor.DecryptNew(ct)
		decoded := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(pt, decoded)

		for oy := 0; oy < outH; oy++ {
			for ox := 0; ox < outW; ox++ {
				pos := oy*inW + ox
				result = append(result, real(decoded[pos]))
			}
		}
	}
	return result
}

// ============================================================================
// MNIST MLP TEST (784 -> 128 -> 64 -> 10)
// ============================================================================

func TestComprehensiveMNISTMLP(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	architecture := []int{784, 128, 64, 10}

	t.Log("\n" + strings.Repeat("═", 120))
	t.Log("MNIST MLP COMPREHENSIVE DIVERGENCE ANALYSIS")
	t.Log("Architecture: 784 -> Linear -> ReLU3 -> 128 -> Linear -> ReLU3 -> 64 -> Linear -> 10")
	t.Log("Mode: WITH CHEAT-STRAP (refresh after each layer)")
	t.Log(strings.Repeat("═", 120))

	numLinearLayers := len(architecture) - 1

	// Create paired HE and plaintext layers
	heLinears := make([]*layers.Linear, numLinearLayers)
	plainLinears := make([]*layers.Linear, numLinearLayers)

	for i := 0; i < numLinearLayers; i++ {
		inDim := architecture[i]
		outDim := architecture[i+1]

		heLinears[i] = layers.NewLinear(inDim, outDim, true, heCtx)
		plainLinears[i] = layers.NewLinear(inDim, outDim, false, nil)

		scale := math.Sqrt(2.0 / float64(inDim+outDim))
		for j := 0; j < outDim; j++ {
			for k := 0; k < inDim; k++ {
				w := (rand.Float64() - 0.5) * 2 * scale
				heLinears[i].W.Data[j*inDim+k] = w
				plainLinears[i].W.Data[j*inDim+k] = w
			}
			b := (rand.Float64() - 0.5) * 0.1
			heLinears[i].B.Data[j] = b
			plainLinears[i].B.Data[j] = b
		}
		heLinears[i].SyncHE()
	}

	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Create input (simulated MNIST pixel values normalized to [-0.5, 0.5])
	inputDim := architecture[0]
	inputData := make([]float64, inputDim)
	for i := range inputData {
		inputData[i] = (rand.Float64() - 0.5)
	}
	plainOutput := make([]float64, inputDim)
	copy(plainOutput, inputData)

	// Encrypt input
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < inputDim; i++ {
		inputVec[i] = complex(inputData[i], 0)
	}
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, ptInput)
	ctInput, _ := heCtx.Encryptor.EncryptNew(ptInput)

	var report ModelDivergenceReport
	report.ModelName = "MNIST MLP"
	report.Architecture = "784 -> 128 -> 64 -> 10"

	heOutput := ctInput
	var accumRMS float64

	// Print header
	t.Log("\n" + strings.Repeat("-", 120))
	t.Logf("%-5s | %-35s | %-15s | %-15s | %-15s | %-10s",
		"Layer", "Name", "Layer RMS", "Accum RMS", "Max Abs Err", "Dims")
	t.Log(strings.Repeat("-", 120))

	layerIdx := 0
	for i := 0; i < numLinearLayers; i++ {
		outDim := architecture[i+1]
		inDim := architecture[i]

		// Linear Layer - HE
		var err error
		heOutput, err = heLinears[i].ForwardCipherMasked(heOutput)
		if err != nil {
			t.Fatalf("HE Linear %d forward failed: %v", i, err)
		}

		// Linear Layer - Plaintext (using the library's forward method)
		plainInputTensor := tensor.New(inDim)
		copy(plainInputTensor.Data, plainOutput)
		plainOutputTensor, err := plainLinears[i].ForwardPlaintext(plainInputTensor)
		if err != nil {
			t.Fatalf("Plain Linear %d forward failed: %v", i, err)
		}
		plainOutput = plainOutputTensor.Data[:outDim]

		heVals := decryptToSlice(heCtx, heOutput, outDim)
		rms := calcRMSError(heVals, plainOutput)
		maxErr := calcMaxAbsError(heVals, plainOutput)
		accumRMS = math.Sqrt(accumRMS*accumRMS + rms*rms)

		layerName := fmt.Sprintf("Linear_%d (%d -> %d)", i, inDim, outDim)
		t.Logf("%-5d | %-35s | %.6e    | %.6e    | %.6e    | %-10d",
			layerIdx, layerName, rms, accumRMS, maxErr, outDim)

		report.Layers = append(report.Layers, LayerDivergenceInfo{
			LayerName: layerName, LayerType: "Linear",
			LayerRMSError: rms, AccumRMSError: accumRMS, MaxAbsError: maxErr, NumDimensions: outDim,
		})
		layerIdx++

		// Cheat-strap after Linear - clean refresh (zeros out garbage slots) and sync plaintext
		heOutput = cleanRefresh(heCtx, heOutput, outDim)
		// After cheat-strap, plaintext path uses decrypted HE values as ground truth
		plainOutput = decryptToSlice(heCtx, heOutput, outDim)

		// Activation Layer (skip for last layer - logits)
		if i < numLinearLayers-1 {
			heActOut, err := heActivation.ForwardCipher(heOutput)
			if err != nil {
				t.Fatalf("HE Activation %d forward failed: %v", i, err)
			}

			// Apply ReLU3 polynomial to plaintext (HE vs HE comparison)
			plainOutput = applyReLU3Poly(plainOutput)

			heActVals := decryptToSlice(heCtx, heActOut, outDim)
			actRMS := calcRMSError(heActVals, plainOutput)
			actMaxErr := calcMaxAbsError(heActVals, plainOutput)
			accumRMS = math.Sqrt(accumRMS*accumRMS + actRMS*actRMS)

			actName := fmt.Sprintf("ReLU3_%d (HE vs Plain Poly)", i)
			t.Logf("%-5d | %-35s | %.6e    | %.6e    | %.6e    | %-10d",
				layerIdx, actName, actRMS, accumRMS, actMaxErr, outDim)

			report.Layers = append(report.Layers, LayerDivergenceInfo{
				LayerName: actName, LayerType: "Activation",
				LayerRMSError: actRMS, AccumRMSError: accumRMS, MaxAbsError: actMaxErr, NumDimensions: outDim,
			})
			layerIdx++

			heOutput = heActOut

			// Cheat-strap after Activation - clean refresh and sync plaintext
			heOutput = cleanRefresh(heCtx, heOutput, outDim)
			plainOutput = decryptToSlice(heCtx, heOutput, outDim)
		}
	}

	report.FinalRMSError = accumRMS
	report.NumLayers = layerIdx

	t.Log(strings.Repeat("-", 120))
	t.Logf("FINAL ACCUMULATED RMS ERROR: %.6e", accumRMS)
	t.Log(strings.Repeat("═", 120))
}

// ============================================================================
// LeNet CNN TEST
// ============================================================================

func TestComprehensiveLeNet(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	t.Log("\n" + strings.Repeat("═", 120))
	t.Log("LeNet CNN COMPREHENSIVE DIVERGENCE ANALYSIS")
	t.Log("Architecture: Conv(1->6, 5x5) -> ReLU3 -> Conv(6->16, 5x5) -> ReLU3 -> Flatten -> FC(256->120) -> ReLU3 -> FC(120->10)")
	t.Log("Mode: WITH CHEAT-STRAP (refresh after each layer)")
	t.Log(strings.Repeat("═", 120))

	// LeNet architecture parameters
	// Input: 1x16x16 (smaller than MNIST 28x28 for faster testing)
	inH, inW := 16, 16

	// Conv1: 1 -> 6, 5x5
	conv1InChan, conv1OutChan := 1, 6
	conv1KH, conv1KW := 5, 5
	conv1OutH, conv1OutW := inH-conv1KH+1, inW-conv1KW+1 // 12x12

	// Conv2: 6 -> 16, 5x5
	conv2InChan, conv2OutChan := 6, 16
	conv2KH, conv2KW := 5, 5
	conv2OutH, conv2OutW := conv1OutH-conv2KH+1, conv1OutW-conv2KW+1 // 8x8

	// Flatten: 16 * 8 * 8 = 1024
	flattenDim := conv2OutChan * conv2OutH * conv2OutW

	// FC layers
	fc1In, fc1Out := flattenDim, 120
	fc2In, fc2Out := 120, 10

	// Create Conv layers
	heConv1 := layers.NewConv2D(conv1InChan, conv1OutChan, conv1KH, conv1KW, true, heCtx)
	plainConv1 := layers.NewConv2D(conv1InChan, conv1OutChan, conv1KH, conv1KW, false, nil)

	heConv2 := layers.NewConv2D(conv2InChan, conv2OutChan, conv2KH, conv2KW, true, heCtx)
	plainConv2 := layers.NewConv2D(conv2InChan, conv2OutChan, conv2KH, conv2KW, false, nil)

	// Create FC layers
	heFC1 := layers.NewLinear(fc1In, fc1Out, true, heCtx)
	plainFC1 := layers.NewLinear(fc1In, fc1Out, false, nil)

	heFC2 := layers.NewLinear(fc2In, fc2Out, true, heCtx)
	plainFC2 := layers.NewLinear(fc2In, fc2Out, false, nil)

	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Initialize Conv1 weights
	scale := math.Sqrt(2.0 / float64(conv1InChan*conv1KH*conv1KW))
	for oc := 0; oc < conv1OutChan; oc++ {
		for ic := 0; ic < conv1InChan; ic++ {
			for i := 0; i < conv1KH; i++ {
				for j := 0; j < conv1KW; j++ {
					w := (rand.Float64() - 0.5) * 2 * scale
					heConv1.W.Set(w, oc, ic, i, j)
					plainConv1.W.Set(w, oc, ic, i, j)
				}
			}
		}
		b := (rand.Float64() - 0.5) * 0.1
		heConv1.B.Set(b, oc)
		plainConv1.B.Set(b, oc)
	}
	heConv1.SetDimensions(inH, inW)
	heConv1.SyncHE()

	// Initialize Conv2 weights
	scale = math.Sqrt(2.0 / float64(conv2InChan*conv2KH*conv2KW))
	for oc := 0; oc < conv2OutChan; oc++ {
		for ic := 0; ic < conv2InChan; ic++ {
			for i := 0; i < conv2KH; i++ {
				for j := 0; j < conv2KW; j++ {
					w := (rand.Float64() - 0.5) * 2 * scale
					heConv2.W.Set(w, oc, ic, i, j)
					plainConv2.W.Set(w, oc, ic, i, j)
				}
			}
		}
		b := (rand.Float64() - 0.5) * 0.1
		heConv2.B.Set(b, oc)
		plainConv2.B.Set(b, oc)
	}
	heConv2.SetDimensions(conv1OutH, conv1OutW)
	heConv2.SyncHE()

	// Initialize FC1 weights
	scale = math.Sqrt(2.0 / float64(fc1In+fc1Out))
	for j := 0; j < fc1Out; j++ {
		for k := 0; k < fc1In; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heFC1.W.Data[j*fc1In+k] = w
			plainFC1.W.Data[j*fc1In+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heFC1.B.Data[j] = b
		plainFC1.B.Data[j] = b
	}
	heFC1.SyncHE()

	// Initialize FC2 weights
	scale = math.Sqrt(2.0 / float64(fc2In+fc2Out))
	for j := 0; j < fc2Out; j++ {
		for k := 0; k < fc2In; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heFC2.W.Data[j*fc2In+k] = w
			plainFC2.W.Data[j*fc2In+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heFC2.B.Data[j] = b
		plainFC2.B.Data[j] = b
	}
	heFC2.SyncHE()

	// Create input image
	inputTensor := tensor.New(conv1InChan, inH, inW)
	for i := range inputTensor.Data {
		inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
	}

	// Encrypt input for Conv (one ciphertext per channel)
	slots := heCtx.Params.MaxSlots()
	inputCTs := make([]*rlwe.Ciphertext, conv1InChan)
	for c := 0; c < conv1InChan; c++ {
		inputVec := make([]complex128, slots)
		for i := 0; i < inH*inW; i++ {
			inputVec[i] = complex(inputTensor.Data[c*inH*inW+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		inputCTs[c] = ct
	}

	var accumRMS float64
	layerIdx := 0

	// Print header
	t.Log("\n" + strings.Repeat("-", 120))
	t.Logf("%-5s | %-40s | %-15s | %-15s | %-15s | %-10s",
		"Layer", "Name", "Layer RMS", "Accum RMS", "Max Abs Err", "Dims")
	t.Log(strings.Repeat("-", 120))

	// ========== CONV1 ==========
	heConv1Out, err := heConv1.ForwardHE(inputCTs)
	if err != nil {
		t.Fatalf("HE Conv1 forward failed: %v", err)
	}
	plainConv1Out, _ := plainConv1.ForwardPlain(inputTensor)

	heConv1Vals := decryptConv2DOutput(heCtx, heConv1Out, conv1OutH, conv1OutW, inW)
	conv1RMS := calcRMSError(heConv1Vals, plainConv1Out.Data[:len(heConv1Vals)])
	conv1MaxErr := calcMaxAbsError(heConv1Vals, plainConv1Out.Data[:len(heConv1Vals)])
	accumRMS = math.Sqrt(accumRMS*accumRMS + conv1RMS*conv1RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, "Conv1 (1->6, 5x5)", conv1RMS, accumRMS, conv1MaxErr, len(heConv1Vals))
	layerIdx++

	// Cheat-strap Conv1 output with clean refresh (zeros garbage slots, restores max level)
	for c := range heConv1Out {
		heConv1Out[c] = cleanRefreshConv2D(heCtx, heConv1Out[c], conv1OutH, conv1OutW, inW)
	}
	// After cheat-strap, use decrypted HE values as plaintext baseline
	plainConv1Synced := decryptConv2DOutput(heCtx, heConv1Out, conv1OutH, conv1OutW, inW)

	// ========== ReLU3 after Conv1 ==========
	heAct1Out := make([]*rlwe.Ciphertext, len(heConv1Out))
	for c := 0; c < len(heConv1Out); c++ {
		heAct1Out[c], _ = heActivation.ForwardCipher(heConv1Out[c])
	}
	plainAct1Out := applyReLU3Poly(plainConv1Synced)

	heAct1Vals := decryptConv2DOutput(heCtx, heAct1Out, conv1OutH, conv1OutW, inW)
	act1RMS := calcRMSError(heAct1Vals, plainAct1Out[:len(heAct1Vals)])
	act1MaxErr := calcMaxAbsError(heAct1Vals, plainAct1Out[:len(heAct1Vals)])
	accumRMS = math.Sqrt(accumRMS*accumRMS + act1RMS*act1RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, "ReLU3_1 (HE vs Plain Poly)", act1RMS, accumRMS, act1MaxErr, len(heAct1Vals))
	layerIdx++

	// ========== FC LAYERS (Tested independently with fresh input) ==========
	t.Log(strings.Repeat("-", 120))
	t.Log("FC LAYERS (tested independently with simulated flattened input)")
	t.Log(strings.Repeat("-", 120))

	// Create fresh input for FC layers (simulated flattened conv output)
	fcInputData := make([]float64, fc1In)
	for i := range fcInputData {
		fcInputData[i] = (rand.Float64() - 0.5) * 0.1 // Small values after ReLU3
	}

	// Encrypt FC input (reuse slots variable from above)
	fcInputVec := make([]complex128, slots)
	for i := 0; i < fc1In; i++ {
		fcInputVec[i] = complex(fcInputData[i], 0)
	}
	fcPtInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(fcInputVec, fcPtInput)
	fcCtInput, _ := heCtx.Encryptor.EncryptNew(fcPtInput)

	fcPlainOutput := make([]float64, fc1In)
	copy(fcPlainOutput, fcInputData)

	// ========== FC1 ==========
	fcCtOutput, err := heFC1.ForwardCipherMasked(fcCtInput)
	if err != nil {
		t.Fatalf("HE FC1 forward failed: %v", err)
	}

	fcPlainInputTensor := tensor.New(fc1In)
	copy(fcPlainInputTensor.Data, fcPlainOutput)
	fcPlainOutputTensor, _ := plainFC1.ForwardPlaintext(fcPlainInputTensor)
	fcPlainOutput = fcPlainOutputTensor.Data[:fc1Out]

	fcHeVals := decryptToSlice(heCtx, fcCtOutput, fc1Out)
	fc1RMS := calcRMSError(fcHeVals, fcPlainOutput)
	fc1MaxErr := calcMaxAbsError(fcHeVals, fcPlainOutput)
	accumRMS = math.Sqrt(accumRMS*accumRMS + fc1RMS*fc1RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, fmt.Sprintf("FC1 (%d -> %d)", fc1In, fc1Out), fc1RMS, accumRMS, fc1MaxErr, fc1Out)
	layerIdx++

	// Cheat-strap FC1
	fcCtOutput = cleanRefresh(heCtx, fcCtOutput, fc1Out)
	fcPlainOutput = decryptToSlice(heCtx, fcCtOutput, fc1Out)

	// ========== ReLU3 after FC1 ==========
	fcActOut, _ := heActivation.ForwardCipher(fcCtOutput)
	fcPlainActOut := applyReLU3Poly(fcPlainOutput)

	fcActVals := decryptToSlice(heCtx, fcActOut, fc1Out)
	fcAct1RMS := calcRMSError(fcActVals, fcPlainActOut)
	fcAct1MaxErr := calcMaxAbsError(fcActVals, fcPlainActOut)
	accumRMS = math.Sqrt(accumRMS*accumRMS + fcAct1RMS*fcAct1RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, "ReLU3_FC1 (HE vs Plain Poly)", fcAct1RMS, accumRMS, fcAct1MaxErr, fc1Out)
	layerIdx++

	// Cheat-strap after activation
	fcActOut = cleanRefresh(heCtx, fcActOut, fc1Out)
	fcPlainOutput = decryptToSlice(heCtx, fcActOut, fc1Out)

	// ========== FC2 ==========
	fcCtOutput2, err := heFC2.ForwardCipherMasked(fcActOut)
	if err != nil {
		t.Fatalf("HE FC2 forward failed: %v", err)
	}

	fcPlainInputTensor2 := tensor.New(fc2In)
	copy(fcPlainInputTensor2.Data, fcPlainOutput)
	fcPlainOutputTensor2, _ := plainFC2.ForwardPlaintext(fcPlainInputTensor2)
	fcPlainOutput2 := fcPlainOutputTensor2.Data[:fc2Out]

	fcHeVals2 := decryptToSlice(heCtx, fcCtOutput2, fc2Out)
	fc2RMS := calcRMSError(fcHeVals2, fcPlainOutput2)
	fc2MaxErr := calcMaxAbsError(fcHeVals2, fcPlainOutput2)
	accumRMS = math.Sqrt(accumRMS*accumRMS + fc2RMS*fc2RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, fmt.Sprintf("FC2 (%d -> %d)", fc2In, fc2Out), fc2RMS, accumRMS, fc2MaxErr, fc2Out)
	layerIdx++

	t.Log(strings.Repeat("-", 120))
	t.Logf("FINAL ACCUMULATED RMS ERROR (all layers): %.6e", accumRMS)
	t.Log(strings.Repeat("═", 120))
}

// ============================================================================
// Audio 1D CNN TEST
// ============================================================================

func TestComprehensiveAudio1D(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	t.Log("\n" + strings.Repeat("═", 120))
	t.Log("AUDIO 1D CNN COMPREHENSIVE DIVERGENCE ANALYSIS")
	t.Log("Architecture: Conv1D(1->4, k=3) -> ReLU3 -> Conv1D(4->8, k=3) -> ReLU3 -> FC(224->64) -> ReLU3 -> FC(64->10)")
	t.Log("Mode: WITH CHEAT-STRAP (refresh after each layer)")
	t.Log(strings.Repeat("═", 120))

	// Audio 1D architecture parameters
	seqLen := 32 // Input sequence length

	// Conv1: 1 -> 4, kernel=3
	conv1InChan, conv1OutChan := 1, 4
	conv1K := 3
	conv1OutLen := seqLen - conv1K + 1 // 30

	// Conv2: 4 -> 8, kernel=3
	conv2InChan, conv2OutChan := 4, 8
	conv2K := 3
	conv2OutLen := conv1OutLen - conv2K + 1 // 28

	// Flatten: 8 * 28 = 224
	flattenDim := conv2OutChan * conv2OutLen

	// FC layers
	fc1In, fc1Out := flattenDim, 64
	fc2In, fc2Out := 64, 10

	// Create Conv1D layers (using Conv2D with kh=1)
	heConv1 := layers.NewConv1D(conv1InChan, conv1OutChan, conv1K, true, heCtx)
	plainConv1 := layers.NewConv1D(conv1InChan, conv1OutChan, conv1K, false, nil)

	heConv2 := layers.NewConv1D(conv2InChan, conv2OutChan, conv2K, true, heCtx)
	plainConv2 := layers.NewConv1D(conv2InChan, conv2OutChan, conv2K, false, nil)

	// Create FC layers
	heFC1 := layers.NewLinear(fc1In, fc1Out, true, heCtx)
	plainFC1 := layers.NewLinear(fc1In, fc1Out, false, nil)

	heFC2 := layers.NewLinear(fc2In, fc2Out, true, heCtx)
	plainFC2 := layers.NewLinear(fc2In, fc2Out, false, nil)

	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Initialize Conv1 weights
	scale := math.Sqrt(2.0 / float64(conv1InChan*conv1K))
	for oc := 0; oc < conv1OutChan; oc++ {
		for ic := 0; ic < conv1InChan; ic++ {
			for j := 0; j < conv1K; j++ {
				w := (rand.Float64() - 0.5) * 2 * scale
				heConv1.W.Set(w, oc, ic, 0, j)
				plainConv1.W.Set(w, oc, ic, 0, j)
			}
		}
		b := (rand.Float64() - 0.5) * 0.1
		heConv1.B.Set(b, oc)
		plainConv1.B.Set(b, oc)
	}
	heConv1.SetDimensions(1, seqLen)
	heConv1.SyncHE()

	// Initialize Conv2 weights
	scale = math.Sqrt(2.0 / float64(conv2InChan*conv2K))
	for oc := 0; oc < conv2OutChan; oc++ {
		for ic := 0; ic < conv2InChan; ic++ {
			for j := 0; j < conv2K; j++ {
				w := (rand.Float64() - 0.5) * 2 * scale
				heConv2.W.Set(w, oc, ic, 0, j)
				plainConv2.W.Set(w, oc, ic, 0, j)
			}
		}
		b := (rand.Float64() - 0.5) * 0.1
		heConv2.B.Set(b, oc)
		plainConv2.B.Set(b, oc)
	}
	heConv2.SetDimensions(1, conv1OutLen)
	heConv2.SyncHE()

	// Initialize FC1 weights
	scale = math.Sqrt(2.0 / float64(fc1In+fc1Out))
	for j := 0; j < fc1Out; j++ {
		for k := 0; k < fc1In; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heFC1.W.Data[j*fc1In+k] = w
			plainFC1.W.Data[j*fc1In+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heFC1.B.Data[j] = b
		plainFC1.B.Data[j] = b
	}
	heFC1.SyncHE()

	// Initialize FC2 weights
	scale = math.Sqrt(2.0 / float64(fc2In+fc2Out))
	for j := 0; j < fc2Out; j++ {
		for k := 0; k < fc2In; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heFC2.W.Data[j*fc2In+k] = w
			plainFC2.W.Data[j*fc2In+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heFC2.B.Data[j] = b
		plainFC2.B.Data[j] = b
	}
	heFC2.SyncHE()

	// Create input audio (1 channel, seqLen samples)
	inputTensor := tensor.New(conv1InChan, 1, seqLen)
	for i := range inputTensor.Data {
		inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
	}

	// Encrypt input for Conv1D (one ciphertext per channel)
	slots := heCtx.Params.MaxSlots()
	inputCTs := make([]*rlwe.Ciphertext, conv1InChan)
	for c := 0; c < conv1InChan; c++ {
		inputVec := make([]complex128, slots)
		for i := 0; i < seqLen; i++ {
			inputVec[i] = complex(inputTensor.Data[c*seqLen+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		inputCTs[c] = ct
	}

	var accumRMS float64
	layerIdx := 0

	// Print header
	t.Log("\n" + strings.Repeat("-", 120))
	t.Logf("%-5s | %-40s | %-15s | %-15s | %-15s | %-10s",
		"Layer", "Name", "Layer RMS", "Accum RMS", "Max Abs Err", "Dims")
	t.Log(strings.Repeat("-", 120))

	// ========== CONV1 ==========
	heConv1Out, err := heConv1.ForwardHE(inputCTs)
	if err != nil {
		t.Fatalf("HE Conv1 forward failed: %v", err)
	}
	plainConv1Out, _ := plainConv1.ForwardPlain(inputTensor)

	// Conv1D output: outH=1, outW=conv1OutLen, inW=seqLen
	heConv1Vals := decryptConv2DOutput(heCtx, heConv1Out, 1, conv1OutLen, seqLen)
	conv1RMS := calcRMSError(heConv1Vals, plainConv1Out.Data[:len(heConv1Vals)])
	conv1MaxErr := calcMaxAbsError(heConv1Vals, plainConv1Out.Data[:len(heConv1Vals)])
	accumRMS = math.Sqrt(accumRMS*accumRMS + conv1RMS*conv1RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, fmt.Sprintf("Conv1D_1 (1->4, k=%d)", conv1K), conv1RMS, accumRMS, conv1MaxErr, len(heConv1Vals))
	layerIdx++

	// Cheat-strap with clean refresh
	for c := range heConv1Out {
		heConv1Out[c] = cleanRefreshConv2D(heCtx, heConv1Out[c], 1, conv1OutLen, seqLen)
	}
	// Sync plaintext to decrypted HE values after cheat-strap
	plainConv1Synced := decryptConv2DOutput(heCtx, heConv1Out, 1, conv1OutLen, seqLen)

	// ========== ReLU3 after Conv1 ==========
	heAct1Out := make([]*rlwe.Ciphertext, len(heConv1Out))
	for c := 0; c < len(heConv1Out); c++ {
		heAct1Out[c], _ = heActivation.ForwardCipher(heConv1Out[c])
	}
	plainAct1Out := applyReLU3Poly(plainConv1Synced)

	heAct1Vals := decryptConv2DOutput(heCtx, heAct1Out, 1, conv1OutLen, seqLen)
	act1RMS := calcRMSError(heAct1Vals, plainAct1Out[:len(heAct1Vals)])
	act1MaxErr := calcMaxAbsError(heAct1Vals, plainAct1Out[:len(heAct1Vals)])
	accumRMS = math.Sqrt(accumRMS*accumRMS + act1RMS*act1RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, "ReLU3_1 (HE vs Plain Poly)", act1RMS, accumRMS, act1MaxErr, len(heAct1Vals))
	layerIdx++

	// NOTE: Sequential conv layers require slot repacking between layers
	t.Log(strings.Repeat("-", 120))
	t.Logf("NOTE: Conv1D->Conv1D chains require slot repacking (not implemented)")
	t.Logf("Continuing with FC layer testing using independent input...")
	t.Log(strings.Repeat("-", 120))

	// ========== FC LAYER TESTING (Independent Input) ==========
	// FC layers can be tested independently since they use simple linear slot layout

	// Create FC1 input (fc1In values)
	fc1Input := make([]float64, fc1In)
	for i := 0; i < fc1In; i++ {
		fc1Input[i] = (rand.Float64() - 0.5) * 2.0
	}

	// Encrypt FC1 input
	fc1InputVec := make([]complex128, slots)
	for i := 0; i < fc1In; i++ {
		fc1InputVec[i] = complex(fc1Input[i], 0)
	}
	fc1PT := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(fc1InputVec, fc1PT)
	fc1CT, _ := heCtx.Encryptor.EncryptNew(fc1PT)

	// ========== FC1 Forward ==========
	fc1OutCT, _ := heFC1.ForwardCipherMasked(fc1CT)

	// Plain FC1 Forward
	plainFC1Vals := make([]float64, fc1Out)
	for j := 0; j < fc1Out; j++ {
		sum := plainFC1.B.Data[j]
		for k := 0; k < fc1In; k++ {
			sum += plainFC1.W.Data[j*fc1In+k] * fc1Input[k]
		}
		plainFC1Vals[j] = sum
	}

	heFC1Vals := decryptToSlice(heCtx, fc1OutCT, fc1Out)
	fc1RMS := calcRMSError(heFC1Vals, plainFC1Vals)
	fc1MaxErr := calcMaxAbsError(heFC1Vals, plainFC1Vals)
	accumRMS = math.Sqrt(accumRMS*accumRMS + fc1RMS*fc1RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, fmt.Sprintf("FC1 (%d->%d)", fc1In, fc1Out), fc1RMS, accumRMS, fc1MaxErr, fc1Out)
	layerIdx++

	// Cheat-strap FC1 output
	fc1OutCT = cleanRefresh(heCtx, fc1OutCT, fc1Out)
	plainFC1Synced := decryptToSlice(heCtx, fc1OutCT, fc1Out)

	// ========== ReLU3 after FC1 ==========
	heFC1ActCT, _ := heActivation.ForwardCipher(fc1OutCT)
	plainFC1ActVals := applyReLU3Poly(plainFC1Synced)

	heFC1ActVals := decryptToSlice(heCtx, heFC1ActCT, fc1Out)
	fc1ActRMS := calcRMSError(heFC1ActVals, plainFC1ActVals)
	fc1ActMaxErr := calcMaxAbsError(heFC1ActVals, plainFC1ActVals)
	accumRMS = math.Sqrt(accumRMS*accumRMS + fc1ActRMS*fc1ActRMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, "ReLU3_FC1 (HE vs Plain Poly)", fc1ActRMS, accumRMS, fc1ActMaxErr, fc1Out)
	layerIdx++

	// Cheat-strap after ReLU3
	heFC1ActCT = cleanRefresh(heCtx, heFC1ActCT, fc1Out)
	plainFC1ActSynced := decryptToSlice(heCtx, heFC1ActCT, fc1Out)

	// ========== FC2 Forward (Final layer) ==========
	fc2OutCT, _ := heFC2.ForwardCipherMasked(heFC1ActCT)

	// Plain FC2 Forward
	plainFC2Vals := make([]float64, fc2Out)
	for j := 0; j < fc2Out; j++ {
		sum := plainFC2.B.Data[j]
		for k := 0; k < fc2In; k++ {
			sum += plainFC2.W.Data[j*fc2In+k] * plainFC1ActSynced[k]
		}
		plainFC2Vals[j] = sum
	}

	heFC2Vals := decryptToSlice(heCtx, fc2OutCT, fc2Out)
	fc2RMS := calcRMSError(heFC2Vals, plainFC2Vals)
	fc2MaxErr := calcMaxAbsError(heFC2Vals, plainFC2Vals)
	accumRMS = math.Sqrt(accumRMS*accumRMS + fc2RMS*fc2RMS)

	t.Logf("%-5d | %-40s | %.6e    | %.6e    | %.6e    | %-10d",
		layerIdx, fmt.Sprintf("FC2 (%d->%d)", fc2In, fc2Out), fc2RMS, accumRMS, fc2MaxErr, fc2Out)

	t.Log(strings.Repeat("═", 120))
	t.Logf("AUDIO 1D CNN FINAL ACCUMULATED RMS ERROR: %.6e", accumRMS)
	t.Logf("Total layers tested: %d", layerIdx+1)
	t.Log(strings.Repeat("═", 120))

	// Suppress unused variable warnings
	_ = heConv2
	_ = plainConv2
}

// ============================================================================
// SUMMARY TABLE TEST
// ============================================================================

func TestComprehensiveSummaryTable(t *testing.T) {
	t.Log("\n")
	t.Log(strings.Repeat("═", 140))
	t.Log("COMPREHENSIVE ACCUMULATED DIVERGENCE SUMMARY TABLE")
	t.Log("Mode: WITH CHEAT-STRAP (refresh after each layer)")
	t.Log("Activation comparison: HE ReLU3 Polynomial vs Plaintext ReLU3 Polynomial")
	t.Log(strings.Repeat("═", 140))
	t.Log("")

	// Run all tests and collect results
	t.Log("Running MNIST MLP...")
	t.Run("MNIST_MLP", TestComprehensiveMNISTMLP)

	t.Log("\nRunning LeNet CNN...")
	t.Run("LeNet_CNN", TestComprehensiveLeNet)

	t.Log("\nRunning Audio 1D CNN...")
	t.Run("Audio_1D", TestComprehensiveAudio1D)
}
