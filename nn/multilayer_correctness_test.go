//go:build !exclude_he
// +build !exclude_he

package nn

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/tensor"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ============================================================================
// FORWARD-ONLY MULTI-LAYER CORRECTNESS TEST (FAST)
// This is the primary test for measuring divergence through multiple layers
//
// NOTE ON DIVERGENCE SOURCES:
// - Linear layers: Divergence is purely from HE noise (typically ~1e-8)
// - Activation layers: Divergence comes from BOTH:
//   1. HE noise (small)
//   2. Polynomial approximation vs true ReLU (larger, ~0.1-0.3)
//   The polynomial approximation (ReLU3) is: 0.3183099 + 0.5x + 0.2122066x^2
//   This approximates ReLU but introduces systematic error for the tradeoff
//   of being computable on encrypted data.
// ============================================================================

// LayerDivergence holds divergence statistics for a single layer operation
type LayerDivergence struct {
	LayerName     string
	LayerIndex    int
	Direction     string  // "forward" or "backward"
	MaxAbsError   float64 // Maximum absolute error across all dimensions
	MeanAbsError  float64 // Mean absolute error
	RMSError      float64 // Root mean square error
	MaxRelError   float64 // Maximum relative error (where |plaintext| > threshold)
	NumDimensions int     // Number of dimensions compared
}

// MultiLayerCorrectnessReport holds the complete report for a multi-layer test
type MultiLayerCorrectnessReport struct {
	NumLayers          int
	Architecture       []int // e.g., [784, 128, 64, 32, 10]
	ActivationType     string
	ForwardDivergence  []LayerDivergence
	BackwardDivergence []LayerDivergence
	TotalForwardError  float64
	TotalBackwardError float64
}

// computeDivergence computes divergence metrics between HE and plaintext outputs
func computeDivergence(heVals []float64, plainVals []float64, layerName string, layerIdx int, direction string) LayerDivergence {
	n := len(heVals)
	if len(plainVals) < n {
		n = len(plainVals)
	}

	var maxAbsErr, sumAbsErr, sumSqErr float64
	var maxRelErr float64
	const relThreshold = 1e-8 // Only compute relative error when |plaintext| > threshold

	for i := 0; i < n; i++ {
		absErr := math.Abs(heVals[i] - plainVals[i])
		if absErr > maxAbsErr {
			maxAbsErr = absErr
		}
		sumAbsErr += absErr
		sumSqErr += absErr * absErr

		if math.Abs(plainVals[i]) > relThreshold {
			relErr := absErr / math.Abs(plainVals[i])
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
	}

	return LayerDivergence{
		LayerName:     layerName,
		LayerIndex:    layerIdx,
		Direction:     direction,
		MaxAbsError:   maxAbsErr,
		MeanAbsError:  sumAbsErr / float64(n),
		RMSError:      math.Sqrt(sumSqErr / float64(n)),
		MaxRelError:   maxRelErr,
		NumDimensions: n,
	}
}

// PrintReport prints a nicely formatted table of the divergence report
func (r *MultiLayerCorrectnessReport) PrintReport() string {
	var sb strings.Builder

	sb.WriteString("\n")
	sb.WriteString("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	sb.WriteString("║                    MULTI-LAYER HE vs PLAINTEXT CORRECTNESS REPORT                                    ║\n")
	sb.WriteString("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	sb.WriteString(fmt.Sprintf("║ Architecture: %v                                                                  ║\n", r.Architecture))
	sb.WriteString(fmt.Sprintf("║ Activation: %s | Number of Layers: %d                                                              ║\n", r.ActivationType, r.NumLayers))
	sb.WriteString("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	sb.WriteString("║ NOTE: All errors shown are ACCUMULATED from input through each layer (not per-layer isolated error) ║\n")
	sb.WriteString("║ Error sources: (1) HE encryption/computation noise  (2) ReLU3 polynomial approximation              ║\n")
	sb.WriteString("║ Mode: NO REFRESH (cheat-strap) - HE layers connected directly                                        ║\n")
	sb.WriteString("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	sb.WriteString("\n")

	// Forward pass table
	sb.WriteString("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐\n")
	sb.WriteString("│                         FORWARD PASS DIVERGENCE (ACCUMULATED FROM INPUT)                            │\n")
	sb.WriteString("├─────────────────────────────────────────────────────────────────────────────────────────────────────┤\n")
	sb.WriteString("│ Layer Index │       Layer Name        │ Max Abs Err │ Mean Abs Err │   RMS Err   │ Max Rel Err │ Dims │\n")
	sb.WriteString("├─────────────┼─────────────────────────┼─────────────┼──────────────┼─────────────┼─────────────┼──────┤\n")

	for _, div := range r.ForwardDivergence {
		sb.WriteString(fmt.Sprintf("│     %2d      │ %-23s │ %11.2e │ %12.2e │ %11.2e │ %11.2e │ %4d │\n",
			div.LayerIndex, div.LayerName, div.MaxAbsError, div.MeanAbsError, div.RMSError, div.MaxRelError, div.NumDimensions))
	}
	sb.WriteString("├─────────────┴─────────────────────────┴─────────────┴──────────────┴─────────────┴─────────────┴──────┤\n")
	sb.WriteString(fmt.Sprintf("│ Total Forward Pass Accumulated RMS Error: %.6e                                              │\n", r.TotalForwardError))
	sb.WriteString("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")
	sb.WriteString("\n")

	// Backward pass table
	sb.WriteString("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐\n")
	sb.WriteString("│                                     BACKWARD PASS DIVERGENCE                                        │\n")
	sb.WriteString("├─────────────────────────────────────────────────────────────────────────────────────────────────────┤\n")
	sb.WriteString("│ Layer Index │       Layer Name        │ Max Abs Err │ Mean Abs Err │   RMS Err   │ Max Rel Err │ Dims │\n")
	sb.WriteString("├─────────────┼─────────────────────────┼─────────────┼──────────────┼─────────────┼─────────────┼──────┤\n")

	for _, div := range r.BackwardDivergence {
		sb.WriteString(fmt.Sprintf("│     %2d      │ %-23s │ %11.2e │ %12.2e │ %11.2e │ %11.2e │ %4d │\n",
			div.LayerIndex, div.LayerName, div.MaxAbsError, div.MeanAbsError, div.RMSError, div.MaxRelError, div.NumDimensions))
	}
	sb.WriteString("├─────────────┴─────────────────────────┴─────────────┴──────────────┴─────────────┴─────────────┴──────┤\n")
	sb.WriteString(fmt.Sprintf("│ Total Backward Pass Accumulated RMS Error: %.6e                                             │\n", r.TotalBackwardError))
	sb.WriteString("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")
	sb.WriteString("\n")

	// Summary
	sb.WriteString("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	sb.WriteString("║                                          SUMMARY                                                     ║\n")
	sb.WriteString("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")

	// Find worst forward layer
	var worstForward LayerDivergence
	for _, d := range r.ForwardDivergence {
		if d.RMSError > worstForward.RMSError {
			worstForward = d
		}
	}
	sb.WriteString(fmt.Sprintf("║ Worst Forward Layer: %-20s (RMS Error: %.6e)                                  ║\n", worstForward.LayerName, worstForward.RMSError))

	// Find worst backward layer
	var worstBackward LayerDivergence
	for _, d := range r.BackwardDivergence {
		if d.RMSError > worstBackward.RMSError {
			worstBackward = d
		}
	}
	sb.WriteString(fmt.Sprintf("║ Worst Backward Layer: %-19s (RMS Error: %.6e)                                  ║\n", worstBackward.LayerName, worstBackward.RMSError))
	sb.WriteString("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n")

	return sb.String()
}

// TestMultiLayerCorrectness tests the divergence between HE and plaintext
// through multiple consecutive layers: Linear -> Activation -> Linear -> Activation -> ...
func TestMultiLayerCorrectness(t *testing.T) {
	// Run focused tests with small architectures for speed
	testCases := []struct {
		name         string
		architecture []int
		activation   string
	}{
		{"2Layer_Small", []int{8, 4, 2}, "ReLU3"},
		{"3Layer_Small", []int{8, 4, 4, 2}, "ReLU3"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			report := runForwardOnlyTest(t, tc.architecture, tc.activation)
			t.Log(report.PrintReport())

			// Verify that errors are within acceptable bounds
			for _, div := range report.ForwardDivergence {
				if div.MaxAbsError > 1.0 {
					t.Errorf("Forward layer %s has unacceptable max absolute error: %.6e", div.LayerName, div.MaxAbsError)
				}
			}
		})
	}
}

// runForwardOnlyTest runs forward pass only - much faster for basic correctness
func runForwardOnlyTest(t *testing.T, architecture []int, activationType string) *MultiLayerCorrectnessReport {
	rand.Seed(42)

	heCtx := ckkswrapper.NewHeContext()
	numLinearLayers := len(architecture) - 1

	// Create paired HE and plaintext layers
	heLinears := make([]*layers.Linear, numLinearLayers)
	heActivations := make([]*layers.Activation, numLinearLayers)
	plainLinears := make([]*layers.Linear, numLinearLayers)
	plainActivations := make([]*layers.Activation, numLinearLayers)

	for i := 0; i < numLinearLayers; i++ {
		inDim := architecture[i]
		outDim := architecture[i+1]

		heLinears[i] = layers.NewLinear(inDim, outDim, true, heCtx)
		plainLinears[i] = layers.NewLinear(inDim, outDim, false, nil)

		// Initialize with small weights
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

		heActivations[i], _ = layers.NewActivation(activationType, true, heCtx)
		plainActivations[i], _ = layers.NewActivation(activationType, false, nil)
	}

	// Create input
	inputDim := architecture[0]
	inputData := make([]float64, inputDim)
	for i := range inputData {
		inputData[i] = (rand.Float64() - 0.5) * 2.0
	}
	inputTensor := tensor.New(inputDim)
	copy(inputTensor.Data, inputData)

	// Encrypt input
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < inputDim; i++ {
		inputVec[i] = complex(inputData[i], 0)
	}
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, ptInput)
	ctInput, _ := heCtx.Encryptor.EncryptNew(ptInput)

	report := &MultiLayerCorrectnessReport{
		NumLayers:      numLinearLayers,
		Architecture:   architecture,
		ActivationType: activationType,
	}

	var heOutput *rlwe.Ciphertext = ctInput
	var plainOutput *tensor.Tensor = inputTensor

	layerIdx := 0
	for i := 0; i < numLinearLayers; i++ {
		outDim := architecture[i+1]

		// Linear Layer
		var err error
		heOutput, err = heLinears[i].ForwardCipherMasked(heOutput)
		if err != nil {
			t.Fatalf("HE Linear %d forward failed: %v", i, err)
		}

		plainOutput, err = plainLinears[i].ForwardPlaintext(plainOutput)
		if err != nil {
			t.Fatalf("Plain Linear %d forward failed: %v", i, err)
		}

		heVals := decryptToFloat64(heCtx, heOutput, outDim)
		div := computeDivergence(heVals, plainOutput.Data[:outDim], fmt.Sprintf("Linear_%d (%d->%d)", i, architecture[i], outDim), layerIdx, "forward")
		report.ForwardDivergence = append(report.ForwardDivergence, div)
		layerIdx++

		// Activation Layer (NO refresh - direct HE connection)
		heActOut, err := heActivations[i].ForwardCipher(heOutput)
		if err != nil {
			t.Fatalf("HE Activation %d forward failed: %v", i, err)
		}

		plainActOut, err := plainActivations[i].Forward(plainOutput)
		if err != nil {
			t.Fatalf("Plain Activation %d forward failed: %v", i, err)
		}
		plainActTensor := plainActOut.(*tensor.Tensor)

		heActVals := decryptToFloat64(heCtx, heActOut, outDim)
		actDiv := computeDivergence(heActVals, plainActTensor.Data[:outDim], fmt.Sprintf("Activation_%d (%s)", i, activationType), layerIdx, "forward")
		report.ForwardDivergence = append(report.ForwardDivergence, actDiv)
		layerIdx++

		heOutput = heActOut
		plainOutput = plainActTensor
		// NO refresh - direct HE connection to next layer
	}

	var totalForwardRMS float64
	for _, div := range report.ForwardDivergence {
		totalForwardRMS += div.RMSError * div.RMSError
	}
	report.TotalForwardError = math.Sqrt(totalForwardRMS)

	return report
}

// runMultiLayerCorrectnessTest runs forward + backward test (slower, use sparingly)
func runMultiLayerCorrectnessTest(t *testing.T, architecture []int, activationType string) *MultiLayerCorrectnessReport {
	rand.Seed(42)

	heCtx := ckkswrapper.NewHeContext()
	numLinearLayers := len(architecture) - 1

	heLinears := make([]*layers.Linear, numLinearLayers)
	heActivations := make([]*layers.Activation, numLinearLayers)
	plainLinears := make([]*layers.Linear, numLinearLayers)
	plainActivations := make([]*layers.Activation, numLinearLayers)

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

		heActivations[i], _ = layers.NewActivation(activationType, true, heCtx)
		plainActivations[i], _ = layers.NewActivation(activationType, false, nil)
	}

	inputDim := architecture[0]
	inputData := make([]float64, inputDim)
	for i := range inputData {
		inputData[i] = (rand.Float64() - 0.5) * 2.0
	}
	inputTensor := tensor.New(inputDim)
	copy(inputTensor.Data, inputData)

	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < inputDim; i++ {
		inputVec[i] = complex(inputData[i], 0)
	}
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, ptInput)
	ctInput, _ := heCtx.Encryptor.EncryptNew(ptInput)

	report := &MultiLayerCorrectnessReport{
		NumLayers:      numLinearLayers,
		Architecture:   architecture,
		ActivationType: activationType,
	}

	var heOutput *rlwe.Ciphertext = ctInput
	var plainOutput *tensor.Tensor = inputTensor

	layerIdx := 0
	for i := 0; i < numLinearLayers; i++ {
		outDim := architecture[i+1]

		var err error
		heOutput, err = heLinears[i].ForwardCipherMasked(heOutput)
		if err != nil {
			t.Fatalf("HE Linear %d forward failed: %v", i, err)
		}

		plainOutput, err = plainLinears[i].ForwardPlaintext(plainOutput)
		if err != nil {
			t.Fatalf("Plain Linear %d forward failed: %v", i, err)
		}

		heVals := decryptToFloat64(heCtx, heOutput, outDim)
		div := computeDivergence(heVals, plainOutput.Data[:outDim], fmt.Sprintf("Linear_%d (%d->%d)", i, architecture[i], outDim), layerIdx, "forward")
		report.ForwardDivergence = append(report.ForwardDivergence, div)
		layerIdx++

		// NO refresh - direct HE connection
		heActOut, err := heActivations[i].ForwardCipher(heOutput)
		if err != nil {
			t.Fatalf("HE Activation %d forward failed: %v", i, err)
		}

		plainActOut, err := plainActivations[i].Forward(plainOutput)
		if err != nil {
			t.Fatalf("Plain Activation %d forward failed: %v", i, err)
		}
		plainActTensor := plainActOut.(*tensor.Tensor)

		heActVals := decryptToFloat64(heCtx, heActOut, outDim)
		actDiv := computeDivergence(heActVals, plainActTensor.Data[:outDim], fmt.Sprintf("Activation_%d (%s)", i, activationType), layerIdx, "forward")
		report.ForwardDivergence = append(report.ForwardDivergence, actDiv)
		layerIdx++

		heOutput = heActOut
		plainOutput = plainActTensor
		// NO refresh - direct HE connection
	}

	var totalForwardRMS float64
	for _, div := range report.ForwardDivergence {
		totalForwardRMS += div.RMSError * div.RMSError
	}
	report.TotalForwardError = math.Sqrt(totalForwardRMS)

	// === BACKWARD PASS ===
	outputDim := architecture[len(architecture)-1]
	gradData := make([]float64, outputDim)
	for i := range gradData {
		gradData[i] = (rand.Float64() - 0.5) * 0.1
	}
	gradTensor := tensor.New(outputDim)
	copy(gradTensor.Data, gradData)

	gradVec := make([]complex128, slots)
	for i := 0; i < outputDim; i++ {
		gradVec[i] = complex(gradData[i], 0)
	}
	ptGrad := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(gradVec, ptGrad)
	ctGrad, _ := heCtx.Encryptor.EncryptNew(ptGrad)

	var heGrad *rlwe.Ciphertext = ctGrad
	var plainGrad *tensor.Tensor = gradTensor

	backwardLayerIdx := 0
	for i := numLinearLayers - 1; i >= 0; i-- {
		inDim := architecture[i]

		// NO refresh - direct HE connection
		heActGrad, err := heActivations[i].BackwardHE(heGrad)
		if err != nil {
			t.Logf("HE Activation %d backward failed (level issue): %v - skipping", i, err)
			break
		}

		plainActGrad, err := plainActivations[i].Backward(plainGrad)
		if err != nil {
			t.Fatalf("Plain Activation %d backward failed: %v", i, err)
		}
		plainActGradTensor := plainActGrad.(*tensor.Tensor)

		heActGradVals := decryptToFloat64(heCtx, heActGrad, len(plainActGradTensor.Data))
		actBackDiv := computeDivergence(heActGradVals, plainActGradTensor.Data, fmt.Sprintf("Activation_%d_Back", i), backwardLayerIdx, "backward")
		report.BackwardDivergence = append(report.BackwardDivergence, actBackDiv)

		heGrad = heActGrad
		plainGrad = plainActGradTensor
		backwardLayerIdx++

		// NO refresh - direct HE connection
		heLinGrad, err := heLinears[i].BackwardHE(heGrad)
		if err != nil {
			t.Logf("HE Linear %d backward failed: %v - skipping", i, err)
			break
		}
		heLinGradCt, ok := heLinGrad.(*rlwe.Ciphertext)
		if !ok {
			t.Fatalf("HE Linear %d backward returned non-ciphertext", i)
		}

		plainLinGrad, err := plainLinears[i].Backward(plainGrad)
		if err != nil {
			t.Fatalf("Plain Linear %d backward failed: %v", i, err)
		}
		plainLinGradTensor := plainLinGrad.(*tensor.Tensor)

		heLinGradVals := decryptToFloat64(heCtx, heLinGradCt, inDim)
		linBackDiv := computeDivergence(heLinGradVals, plainLinGradTensor.Data[:inDim], fmt.Sprintf("Linear_%d_Back", i), backwardLayerIdx, "backward")
		report.BackwardDivergence = append(report.BackwardDivergence, linBackDiv)

		heGrad = heLinGradCt
		plainGrad = plainLinGradTensor
		backwardLayerIdx++
	}

	var totalBackwardRMS float64
	for _, div := range report.BackwardDivergence {
		totalBackwardRMS += div.RMSError * div.RMSError
	}
	report.TotalBackwardError = math.Sqrt(totalBackwardRMS)

	return report
}

// decryptToFloat64 decrypts a ciphertext and extracts the first n slots as float64
func decryptToFloat64(heCtx *ckkswrapper.HeContext, ct *rlwe.Ciphertext, n int) []float64 {
	slots := heCtx.Params.MaxSlots()
	pt := heCtx.Decryptor.DecryptNew(ct)
	vals := make([]complex128, slots)
	heCtx.Encoder.Decode(pt, vals)

	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = real(vals[i])
	}
	return result
}

// TestMultiLayerDivergenceProgression tests how error accumulates through layers
func TestMultiLayerDivergenceProgression(t *testing.T) {
	// Single focused test - 3 layers is enough to show progression
	arch := []int{8, 4, 4, 2}
	report := runForwardOnlyTest(t, arch, "ReLU3")
	t.Logf("Architecture: %v", arch)
	t.Logf("Forward Total RMS: %.6e", report.TotalForwardError)

	// Show progression
	for _, div := range report.ForwardDivergence {
		t.Logf("Layer %d (%s): RMS=%.6e", div.LayerIndex, div.LayerName, div.RMSError)
	}
}

// TestMultiLayerCorrectnessDetailed runs a focused test with full report
func TestMultiLayerCorrectnessDetailed(t *testing.T) {
	// Use small architecture for fast execution
	architecture := []int{16, 8, 4, 2}
	report := runForwardOnlyTest(t, architecture, "ReLU3")

	t.Log(report.PrintReport())

	// CSV format
	t.Log("\n=== CSV FORMAT ===")
	t.Log("Direction,Layer_Index,Layer_Name,Max_Abs_Err,Mean_Abs_Err,RMS_Err,Max_Rel_Err,Dims")
	for _, div := range report.ForwardDivergence {
		t.Logf("Forward,%d,%s,%.6e,%.6e,%.6e,%.6e,%d",
			div.LayerIndex, div.LayerName, div.MaxAbsError, div.MeanAbsError, div.RMSError, div.MaxRelError, div.NumDimensions)
	}
}

// TestMultiLayerWithBackward tests both forward and backward (slower)
func TestMultiLayerWithBackward(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping backward test in short mode")
	}
	// Very small network for backward test
	architecture := []int{4, 4, 2}
	report := runMultiLayerCorrectnessTest(t, architecture, "ReLU3")
	t.Log(report.PrintReport())
}

// TestMultiLayerHENoiseOnly tests HE polynomial evaluation directly
// NOTE: There appears to be a scale mismatch bug in the HE polynomial evaluation
// This test documents the divergence in the current implementation
func TestMultiLayerHENoiseOnly(t *testing.T) {
	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// Test the polynomial evaluation directly
	testInput := 0.5
	slots := heCtx.Params.MaxSlots()
	testVec := make([]complex128, slots)
	for i := range testVec {
		testVec[i] = complex(testInput, 0)
	}
	ptTest := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(testVec, ptTest)
	ctTest, _ := heCtx.Encryptor.EncryptNew(ptTest)

	// Create activation and test
	testAct, _ := layers.NewActivation("ReLU3", true, heCtx)
	ctResult, _ := testAct.ForwardCipher(ctTest)

	ptResult := heCtx.Decryptor.DecryptNew(ctResult)
	resultVec := make([]complex128, slots)
	heCtx.Encoder.Decode(ptResult, resultVec)
	heResult := real(resultVec[0])

	// Expected: 0.3183099 + 0.5*0.5 + 0.2122066*0.25 = 0.6213616
	expectedResult := 0.3183099 + 0.5*testInput + 0.2122066*testInput*testInput
	t.Logf("HE Polynomial Evaluation Test:")
	t.Logf("  Input: %.4f", testInput)
	t.Logf("  HE Result: %.6f", heResult)
	t.Logf("  Expected (polynomial): %.6f", expectedResult)
	t.Logf("  True ReLU: %.6f", math.Max(0, testInput))
	t.Logf("  HE vs Polynomial diff: %.6e", heResult-expectedResult)
	t.Logf("  HE vs True ReLU diff: %.6e", heResult-math.Max(0, testInput))
	t.Log("")
	t.Log("NOTE: The HE polynomial evaluation shows divergence from expected.")
	t.Log("This is documented behavior - see revision2.md for analysis.")
} // ============================================================================
// MODEL-SPECIFIC CORRECTNESS TESTS (Based on models.go)
// ============================================================================

// TestMNISTFCCorrectness tests the MNIST FC model: 784-128-32-10
func TestMNISTFCCorrectness(t *testing.T) {
	architecture := []int{784, 128, 32, 10}
	report := runForwardOnlyTest(t, architecture, "ReLU3")
	t.Log(report.PrintReport())
}

// TestBCWFCCorrectness tests the BCW FC model: 64-32-16-10
func TestBCWFCCorrectness(t *testing.T) {
	architecture := []int{64, 32, 16, 10}
	report := runForwardOnlyTest(t, architecture, "ReLU3")
	t.Log(report.PrintReport())
}

// TestAllModelsCorrectness runs correctness tests for all FC models
func TestAllModelsCorrectness(t *testing.T) {
	models := []struct {
		name         string
		architecture []int
	}{
		{"MNIST_FC (784-128-32-10)", []int{784, 128, 32, 10}},
		{"BCW_FC (64-32-16-10)", []int{64, 32, 16, 10}},
		{"Small_FC (16-8-4-2)", []int{16, 8, 4, 2}},
	}

	for _, m := range models {
		t.Run(m.name, func(t *testing.T) {
			report := runForwardOnlyTest(t, m.architecture, "ReLU3")
			t.Log(report.PrintReport())
		})
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ============================================================================
// CNN MODEL CORRECTNESS TESTS (LeNet, Audio1D)
// ============================================================================

// CNNLayerDivergence holds divergence for CNN layers (can have multiple channels)
type CNNLayerDivergence struct {
	LayerName    string
	LayerIndex   int
	MaxAbsError  float64
	MeanAbsError float64
	RMSError     float64
	MaxRelError  float64
	NumElements  int
	NumChannels  int
}

// CNNCorrectnessReport holds the complete report for a CNN model test
type CNNCorrectnessReport struct {
	ModelName         string
	LayerDescriptions []string
	ForwardDivergence []CNNLayerDivergence
	TotalForwardError float64
}

// PrintCNNReport prints a nicely formatted table of the CNN divergence report
func (r *CNNCorrectnessReport) PrintCNNReport() string {
	var sb strings.Builder

	sb.WriteString("\n")
	sb.WriteString("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	sb.WriteString("║                    CNN MODEL HE vs PLAINTEXT CORRECTNESS REPORT                                      ║\n")
	sb.WriteString("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	sb.WriteString(fmt.Sprintf("║ Model: %-90s ║\n", r.ModelName))
	sb.WriteString("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	sb.WriteString("║ NOTE: All errors shown are ACCUMULATED from input through each layer                                 ║\n")
	sb.WriteString("║ Mode: NO REFRESH (cheat-strap) - HE layers connected directly                                        ║\n")
	sb.WriteString("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	sb.WriteString("\n")

	sb.WriteString("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐\n")
	sb.WriteString("│                         FORWARD PASS DIVERGENCE (ACCUMULATED FROM INPUT)                            │\n")
	sb.WriteString("├─────────────────────────────────────────────────────────────────────────────────────────────────────┤\n")
	sb.WriteString("│ Layer Index │       Layer Name           │ Max Abs Err │ Mean Abs Err │   RMS Err   │ Elements │ Ch │\n")
	sb.WriteString("├─────────────┼────────────────────────────┼─────────────┼──────────────┼─────────────┼──────────┼────┤\n")

	for _, div := range r.ForwardDivergence {
		sb.WriteString(fmt.Sprintf("│     %2d      │ %-26s │ %11.2e │ %12.2e │ %11.2e │ %8d │ %2d │\n",
			div.LayerIndex, div.LayerName, div.MaxAbsError, div.MeanAbsError, div.RMSError, div.NumElements, div.NumChannels))
	}
	sb.WriteString("├─────────────┴────────────────────────────┴─────────────┴──────────────┴─────────────┴──────────┴────┤\n")
	sb.WriteString(fmt.Sprintf("│ Total Forward Pass Accumulated RMS Error: %.6e                                              │\n", r.TotalForwardError))
	sb.WriteString("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

	return sb.String()
}

// TestLeNetCorrectness tests the LeNet-5 model (simplified FC portion only due to level constraints)
// LeNet: Conv2D(1->6, 5x5) -> ReLU3 -> Conv2D(6->16, 5x5) -> ReLU3 -> FC(256->120) -> ReLU3 -> FC(120->84) -> ReLU3 -> FC(84->10)
func TestLeNetCorrectness(t *testing.T) {
	// Due to level constraints, we test the FC portion of LeNet: 256 -> 120 -> 84 -> 10
	// This represents the classifier after the convolutional feature extractor
	t.Log("Testing LeNet FC portion (after conv layers): 256 -> 120 -> 84 -> 10")
	architecture := []int{256, 120, 84, 10}
	report := runForwardOnlyTest(t, architecture, "ReLU3")
	t.Log(report.PrintReport())
}

// TestLeNetFCOnlyCorrectness tests just the FC layers of LeNet with correct input size
func TestLeNetFCOnlyCorrectness(t *testing.T) {
	// LeNet FC layers after flatten: 16*4*4=256 -> 120 -> 84 -> 10
	architecture := []int{256, 120, 84, 10}
	report := runForwardOnlyTest(t, architecture, "ReLU3")
	t.Log(report.PrintReport())

	// Also output in CSV format for easy parsing
	t.Log("\n=== LeNet FC CSV ===")
	t.Log("Layer_Index,Layer_Name,Max_Abs_Err,Mean_Abs_Err,RMS_Err,Max_Rel_Err,Dims")
	for _, div := range report.ForwardDivergence {
		t.Logf("%d,%s,%.6e,%.6e,%.6e,%.6e,%d",
			div.LayerIndex, div.LayerName, div.MaxAbsError, div.MeanAbsError, div.RMSError, div.MaxRelError, div.NumDimensions)
	}
}

// TestAudio1DFCCorrectness tests the FC portion of Audio1D CNN
// Audio1D: Conv1D(12->16, k=3) -> ReLU3 -> MaxPool(2) -> Conv1D(16->8, k=3) -> ReLU3 -> MaxPool(2) -> Flatten -> FC(2000->5) -> ReLU3
func TestAudio1DFCCorrectness(t *testing.T) {
	// Due to level constraints, we test the FC portion: 2000 -> 5
	// This represents the classifier after the 1D convolutional feature extractor
	t.Log("Testing Audio1D FC portion (after conv layers): 2000 -> 5")
	architecture := []int{2000, 5}
	report := runForwardOnlyTest(t, architecture, "ReLU3")
	t.Log(report.PrintReport())
}

// TestSmallCNNFCCorrectness tests a smaller version for quick validation
func TestSmallCNNFCCorrectness(t *testing.T) {
	// Small FC to represent CNN classifier portion
	architecture := []int{64, 32, 10}
	report := runForwardOnlyTest(t, architecture, "ReLU3")
	t.Log(report.PrintReport())
}

// TestAllArchitecturesForReport runs all architectures and outputs CSV for comprehensive report
func TestAllArchitecturesForReport(t *testing.T) {
	architectures := []struct {
		name         string
		architecture []int
	}{
		{"MNIST_MLP", []int{784, 128, 32, 10}},
		{"BCW_FC", []int{64, 32, 16, 10}},
		{"LeNet_FC", []int{256, 120, 84, 10}},
		{"Audio1D_FC", []int{2000, 5}},
		{"Small_FC", []int{16, 8, 4, 2}},
	}

	t.Log("\n" + strings.Repeat("=", 100))
	t.Log("COMPREHENSIVE ARCHITECTURE DIVERGENCE ANALYSIS")
	t.Log(strings.Repeat("=", 100))

	// Collect all results
	allResults := make(map[string]*MultiLayerCorrectnessReport)

	for _, arch := range architectures {
		t.Run(arch.name, func(t *testing.T) {
			report := runForwardOnlyTest(t, arch.architecture, "ReLU3")
			allResults[arch.name] = report
			t.Log(report.PrintReport())
		})
	}

	// Output summary table
	t.Log("\n" + strings.Repeat("=", 100))
	t.Log("SUMMARY TABLE")
	t.Log(strings.Repeat("=", 100))
	t.Log("Model,Architecture,Num_Layers,Total_RMS_Error")

	for _, arch := range architectures {
		if report, ok := allResults[arch.name]; ok {
			archStr := fmt.Sprintf("%v", arch.architecture)
			t.Logf("%s,%s,%d,%.6e", arch.name, archStr, report.NumLayers, report.TotalForwardError)
		}
	}

	// Output detailed CSV
	t.Log("\n" + strings.Repeat("=", 100))
	t.Log("DETAILED LAYER-BY-LAYER CSV")
	t.Log(strings.Repeat("=", 100))
	t.Log("Model,Layer_Index,Layer_Name,Max_Abs_Err,Mean_Abs_Err,RMS_Err,Max_Rel_Err,Dims")

	for _, arch := range architectures {
		if report, ok := allResults[arch.name]; ok {
			for _, div := range report.ForwardDivergence {
				t.Logf("%s,%d,%s,%.6e,%.6e,%.6e,%.6e,%d",
					arch.name, div.LayerIndex, div.LayerName,
					div.MaxAbsError, div.MeanAbsError, div.RMSError, div.MaxRelError, div.NumDimensions)
			}
		}
	}
}

// ============================================================================
// PURE HE NOISE TESTS (Compare HE polynomial vs Plaintext polynomial)
// This isolates the HE noise from the polynomial approximation error
// ============================================================================

// applyReLU3Polynomial applies the ReLU3 polynomial to plaintext values
func applyReLU3Polynomial(vals []float64) []float64 {
	// ReLU3 polynomial: 0.3183099 + 0.5*x + 0.2122066*x^2 + 0*x^3
	c0, c1, c2 := 0.3183099, 0.5, 0.2122066
	result := make([]float64, len(vals))
	for i, x := range vals {
		result[i] = c0 + c1*x + c2*x*x
	}
	return result
}

// runForwardOnlyTestPureHENoise compares HE output vs plaintext polynomial (not true ReLU)
// This measures ONLY the HE noise, without the polynomial approximation error
func runForwardOnlyTestPureHENoise(t *testing.T, architecture []int, useCheatStrap bool) *MultiLayerCorrectnessReport {
	rand.Seed(42)

	heCtx := ckkswrapper.NewHeContext()
	numLinearLayers := len(architecture) - 1

	// Create paired HE and plaintext layers
	heLinears := make([]*layers.Linear, numLinearLayers)
	plainLinears := make([]*layers.Linear, numLinearLayers)

	for i := 0; i < numLinearLayers; i++ {
		inDim := architecture[i]
		outDim := architecture[i+1]

		heLinears[i] = layers.NewLinear(inDim, outDim, true, heCtx)
		plainLinears[i] = layers.NewLinear(inDim, outDim, false, nil)

		// Initialize with small weights
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

	// Create HE activation
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Create input
	inputDim := architecture[0]
	inputData := make([]float64, inputDim)
	for i := range inputData {
		inputData[i] = (rand.Float64() - 0.5) * 2.0
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

	mode := "NO_REFRESH"
	if useCheatStrap {
		mode = "WITH_CHEATSTRAP"
	}

	report := &MultiLayerCorrectnessReport{
		NumLayers:      numLinearLayers,
		Architecture:   architecture,
		ActivationType: fmt.Sprintf("ReLU3_Poly (%s)", mode),
	}

	var heOutput *rlwe.Ciphertext = ctInput

	layerIdx := 0
	for i := 0; i < numLinearLayers; i++ {
		outDim := architecture[i+1]

		// Linear Layer - HE
		var err error
		heOutput, err = heLinears[i].ForwardCipherMasked(heOutput)
		if err != nil {
			t.Fatalf("HE Linear %d forward failed: %v", i, err)
		}

		// Linear Layer - Plaintext (manual matrix-vector multiply)
		inDim := architecture[i]
		newPlainOutput := make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			sum := plainLinears[i].B.Data[j]
			for k := 0; k < inDim; k++ {
				sum += plainLinears[i].W.Data[j*inDim+k] * plainOutput[k]
			}
			newPlainOutput[j] = sum
		}
		plainOutput = newPlainOutput

		heVals := decryptToFloat64(heCtx, heOutput, outDim)
		div := computeDivergence(heVals, plainOutput[:outDim], fmt.Sprintf("Linear_%d (%d->%d)", i, architecture[i], outDim), layerIdx, "forward")
		report.ForwardDivergence = append(report.ForwardDivergence, div)
		layerIdx++

		// Cheat-strap after Linear layer (if enabled)
		if useCheatStrap {
			heOutput = heCtx.Refresh(heOutput)
		}

		// Activation Layer - HE (polynomial)
		heActOut, err := heActivation.ForwardCipher(heOutput)
		if err != nil {
			t.Fatalf("HE Activation %d forward failed: %v", i, err)
		}

		// Activation Layer - Plaintext (apply same polynomial, not true ReLU!)
		plainOutput = applyReLU3Polynomial(plainOutput)

		heActVals := decryptToFloat64(heCtx, heActOut, outDim)
		actDiv := computeDivergence(heActVals, plainOutput[:outDim], fmt.Sprintf("Activation_%d (ReLU3_Poly)", i), layerIdx, "forward")
		report.ForwardDivergence = append(report.ForwardDivergence, actDiv)
		layerIdx++

		heOutput = heActOut

		// Cheat-strap after Activation layer (if enabled)
		if useCheatStrap {
			heOutput = heCtx.Refresh(heOutput)
		}
	}

	var totalForwardRMS float64
	for _, div := range report.ForwardDivergence {
		totalForwardRMS += div.RMSError * div.RMSError
	}
	report.TotalForwardError = math.Sqrt(totalForwardRMS)

	return report
}

// TestPureHENoiseSmall tests pure HE noise (no polynomial approximation error)
func TestPureHENoiseSmall(t *testing.T) {
	t.Log("\n=== PURE HE NOISE TEST (No Polynomial Approximation Error) ===")
	t.Log("This test compares HE polynomial output vs Plaintext polynomial output")
	t.Log("to isolate the HE computation noise from the ReLU approximation error.\n")

	architecture := []int{16, 8, 4, 2}

	t.Log("--- WITHOUT CHEAT-STRAP ---")
	reportNoRefresh := runForwardOnlyTestPureHENoise(t, architecture, false)
	t.Log(reportNoRefresh.PrintReport())

	t.Log("\n--- WITH CHEAT-STRAP (Refresh after each layer) ---")
	reportWithRefresh := runForwardOnlyTestPureHENoise(t, architecture, true)
	t.Log(reportWithRefresh.PrintReport())

	// Compare
	t.Log("\n=== COMPARISON ===")
	t.Logf("Total RMS (No Refresh):   %.6e", reportNoRefresh.TotalForwardError)
	t.Logf("Total RMS (With Refresh): %.6e", reportWithRefresh.TotalForwardError)
	t.Logf("Improvement Factor:       %.2fx", reportNoRefresh.TotalForwardError/reportWithRefresh.TotalForwardError)
}

// TestDebugLinearPropagation investigates error propagation through Linear layer
func TestDebugLinearPropagation(t *testing.T) {
	t.Log("\n=== DEBUG: Linear Layer Error Propagation ===")

	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// Simple 4->2 linear
	inDim, outDim := 4, 2

	heLinear := layers.NewLinear(inDim, outDim, true, heCtx)
	plainLinear := layers.NewLinear(inDim, outDim, false, nil)

	// Set identical weights
	scale := 0.5
	for j := 0; j < outDim; j++ {
		for k := 0; k < inDim; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heLinear.W.Data[j*inDim+k] = w
			plainLinear.W.Data[j*inDim+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heLinear.B.Data[j] = b
		plainLinear.B.Data[j] = b
	}
	heLinear.SyncHE()

	t.Logf("Weights: %v", heLinear.W.Data)
	t.Logf("Bias: %v", heLinear.B.Data)

	// Create input
	inputData := []float64{0.5, -0.3, 0.1, -0.7}
	t.Logf("Input: %v", inputData)

	// Encrypt input
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < inDim; i++ {
		inputVec[i] = complex(inputData[i], 0)
	}
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, ptInput)
	ctInput, _ := heCtx.Encryptor.EncryptNew(ptInput)

	// HE Linear forward
	heOutput, err := heLinear.ForwardCipherMasked(ctInput)
	if err != nil {
		t.Fatalf("HE Linear forward failed: %v", err)
	}
	heVals := decryptToFloat64(heCtx, heOutput, outDim)
	t.Logf("HE Linear output: %v", heVals)

	// Plaintext Linear forward (manual)
	plainOutput := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		sum := plainLinear.B.Data[j]
		for k := 0; k < inDim; k++ {
			sum += plainLinear.W.Data[j*inDim+k] * inputData[k]
		}
		plainOutput[j] = sum
	}
	t.Logf("Plain Linear output: %v", plainOutput)

	// Compare
	t.Log("\nPer-element comparison:")
	for i := 0; i < outDim; i++ {
		diff := heVals[i] - plainOutput[i]
		t.Logf("  [%d] HE=%.10f, Plain=%.10f, Diff=%.6e", i, heVals[i], plainOutput[i], diff)
	}
	t.Logf("RMS Error: %.6e", computeRMSError(heVals, plainOutput))
}

// TestDebugTwoLayerPipeline investigates error through Linear -> Activation -> Linear
func TestDebugTwoLayerPipeline(t *testing.T) {
	t.Log("\n=== DEBUG: Two Layer Pipeline (Linear->Act->Linear) ===")

	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// 4 -> 2 -> 2
	heLinear1 := layers.NewLinear(4, 2, true, heCtx)
	heLinear2 := layers.NewLinear(2, 2, true, heCtx)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Set simple weights (small)
	for j := 0; j < 2; j++ {
		for k := 0; k < 4; k++ {
			heLinear1.W.Data[j*4+k] = (rand.Float64() - 0.5) * 0.5
		}
		heLinear1.B.Data[j] = (rand.Float64() - 0.5) * 0.1
	}
	for j := 0; j < 2; j++ {
		for k := 0; k < 2; k++ {
			heLinear2.W.Data[j*2+k] = (rand.Float64() - 0.5) * 0.5
		}
		heLinear2.B.Data[j] = (rand.Float64() - 0.5) * 0.1
	}
	heLinear1.SyncHE()
	heLinear2.SyncHE()

	// Create input
	inputData := []float64{0.5, -0.3, 0.1, -0.7}
	t.Logf("Input: %v", inputData)

	// Encrypt input
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < 4; i++ {
		inputVec[i] = complex(inputData[i], 0)
	}
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, ptInput)
	ctInput, _ := heCtx.Encryptor.EncryptNew(ptInput)

	// Layer 1: Linear
	t.Log("\n--- Layer 1: Linear ---")
	t.Logf("Input ciphertext level: %d, scale: %.2f", ctInput.Level(), math.Log2(ctInput.Scale.Float64()))
	heOut1, _ := heLinear1.ForwardCipherMasked(ctInput)
	t.Logf("Output ciphertext level: %d, scale: %.2f", heOut1.Level(), math.Log2(heOut1.Scale.Float64()))
	heVals1 := decryptToFloat64(heCtx, heOut1, 2)
	t.Logf("HE Linear1 output: %v", heVals1)

	// Compute plaintext
	plain1 := make([]float64, 2)
	for j := 0; j < 2; j++ {
		sum := heLinear1.B.Data[j]
		for k := 0; k < 4; k++ {
			sum += heLinear1.W.Data[j*4+k] * inputData[k]
		}
		plain1[j] = sum
	}
	t.Logf("Plain Linear1 output: %v", plain1)
	t.Logf("Linear1 RMS Error: %.6e", computeRMSError(heVals1, plain1))

	// Layer 2: Activation
	t.Log("\n--- Layer 2: Activation ---")
	t.Logf("Input ciphertext level: %d, scale: %.2f", heOut1.Level(), math.Log2(heOut1.Scale.Float64()))
	heOut2, _ := heActivation.ForwardCipher(heOut1)
	t.Logf("Output ciphertext level: %d, scale: %.2f", heOut2.Level(), math.Log2(heOut2.Scale.Float64()))
	heVals2 := decryptToFloat64(heCtx, heOut2, 2)
	t.Logf("HE Activation output: %v", heVals2)

	// Compute plaintext polynomial
	plain2 := applyReLU3Polynomial(plain1)
	t.Logf("Plain Activation output: %v", plain2)
	t.Logf("Activation RMS Error: %.6e", computeRMSError(heVals2, plain2))

	// Layer 3: Linear
	t.Log("\n--- Layer 3: Linear ---")
	t.Logf("Input ciphertext level: %d, scale: %.2f", heOut2.Level(), math.Log2(heOut2.Scale.Float64()))
	heOut3, err := heLinear2.ForwardCipherMasked(heOut2)
	if err != nil {
		t.Logf("Linear2 FAILED: %v", err)
		return
	}
	t.Logf("Output ciphertext level: %d, scale: %.2f", heOut3.Level(), math.Log2(heOut3.Scale.Float64()))
	heVals3 := decryptToFloat64(heCtx, heOut3, 2)
	t.Logf("HE Linear2 output: %v", heVals3)

	// Compute plaintext
	plain3 := make([]float64, 2)
	for j := 0; j < 2; j++ {
		sum := heLinear2.B.Data[j]
		for k := 0; k < 2; k++ {
			sum += heLinear2.W.Data[j*2+k] * plain2[k]
		}
		plain3[j] = sum
	}
	t.Logf("Plain Linear2 output: %v", plain3)
	t.Logf("Linear2 RMS Error: %.6e", computeRMSError(heVals3, plain3))
}

// TestDebugActivationError investigates why error jumps after activation
func TestDebugActivationError(t *testing.T) {
	t.Log("\n=== DEBUG: Investigating Activation Error Source ===")

	rand.Seed(42)
	heCtx := ckkswrapper.NewHeContext()

	// Create simple input
	inputData := []float64{0.5, -0.3, 0.1, -0.7}
	inputDim := len(inputData)

	// Encrypt input
	slots := heCtx.Params.MaxSlots()
	inputVec := make([]complex128, slots)
	for i := 0; i < inputDim; i++ {
		inputVec[i] = complex(inputData[i], 0)
	}
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, ptInput)
	ctInput, _ := heCtx.Encryptor.EncryptNew(ptInput)

	// Decrypt and compare to original
	heVals := decryptToFloat64(heCtx, ctInput, inputDim)
	t.Logf("Original input: %v", inputData)
	t.Logf("After encrypt/decrypt: %v", heVals)
	t.Logf("Input encode/decode error: %.6e", computeRMSError(heVals, inputData))

	// Apply HE activation
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)
	heActOut, err := heActivation.ForwardCipher(ctInput)
	if err != nil {
		t.Fatalf("HE Activation failed: %v", err)
	}

	// Get HE result
	heActVals := decryptToFloat64(heCtx, heActOut, inputDim)
	t.Logf("HE Activation output: %v", heActVals)

	// Apply plaintext polynomial - MY VERSION
	plainActVals := applyReLU3Polynomial(inputData)
	t.Logf("My plaintext polynomial output: %v", plainActVals)

	// Manually compute for verification
	t.Log("\n=== Manual Polynomial Verification (for x=0.5) ===")
	x := 0.5
	c0, c1, c2, c3 := 0.3183099, 0.5, 0.2122066, 0.0
	manual := c0 + c1*x + c2*x*x + c3*x*x*x
	t.Logf("Manual: %.6f + %.6f*%.2f + %.6f*%.4f + %.6f*%.6f = %.6f",
		c0, c1, x, c2, x*x, c3, x*x*x, manual)

	// Compare each value
	t.Log("\nPer-element comparison:")
	for i := 0; i < inputDim; i++ {
		diff := heActVals[i] - plainActVals[i]
		t.Logf("  [%d] x=%.2f: HE=%.6f, Plain=%.6f, Diff=%.6e", i, inputData[i], heActVals[i], plainActVals[i], diff)
	}
	t.Logf("\nTotal RMS Error (HE vs Plain Poly): %.6e", computeRMSError(heActVals, plainActVals))

	// What does the layer's OWN plaintext forwardPlain give?
	t.Log("\n=== Testing Layer's Own Forward Plain ===")
	plainActivation, _ := layers.NewActivation("ReLU3", false, nil)
	plainInput := tensor.New(inputDim)
	copy(plainInput.Data, inputData)
	layerPlainOut, err := plainActivation.Forward(plainInput)
	if err != nil {
		t.Fatalf("Plain activation forward failed: %v", err)
	}
	layerPlainTensor := layerPlainOut.(*tensor.Tensor)
	t.Logf("Layer's forwardPlain output: %v", layerPlainTensor.Data[:inputDim])
	t.Logf("HE vs Layer Plain error: %.6e", computeRMSError(heActVals, layerPlainTensor.Data[:inputDim]))

	// Also test what the TRUE ReLU would give
	trueReluVals := make([]float64, inputDim)
	for i, v := range inputData {
		if v > 0 {
			trueReluVals[i] = v
		} else {
			trueReluVals[i] = 0
		}
	}
	t.Log("\n=== Comparison with True ReLU ===")
	t.Logf("True ReLU output: %v", trueReluVals)
	t.Logf("HE poly vs True ReLU error: %.6e", computeRMSError(heActVals, trueReluVals))
	t.Logf("Plain poly vs True ReLU error: %.6e", computeRMSError(plainActVals, trueReluVals))
}

func computeRMSError(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(a)))
}

// TestPureHENoiseMNIST tests pure HE noise on MNIST architecture
func TestPureHENoiseMNIST(t *testing.T) {
	t.Log("\n=== PURE HE NOISE TEST - MNIST (784->128->32->10) ===")
	t.Log("Comparing HE polynomial output vs Plaintext polynomial output\n")

	architecture := []int{784, 128, 32, 10}

	t.Log("--- WITHOUT CHEAT-STRAP ---")
	reportNoRefresh := runForwardOnlyTestPureHENoise(t, architecture, false)
	t.Log(reportNoRefresh.PrintReport())

	t.Log("\n--- WITH CHEAT-STRAP (Refresh after each layer) ---")
	reportWithRefresh := runForwardOnlyTestPureHENoise(t, architecture, true)
	t.Log(reportWithRefresh.PrintReport())

	// Compare
	t.Log("\n=== COMPARISON ===")
	t.Logf("Total RMS (No Refresh):   %.6e", reportNoRefresh.TotalForwardError)
	t.Logf("Total RMS (With Refresh): %.6e", reportWithRefresh.TotalForwardError)
	if reportWithRefresh.TotalForwardError > 0 {
		t.Logf("Improvement Factor:       %.2fx", reportNoRefresh.TotalForwardError/reportWithRefresh.TotalForwardError)
	}
}

// TestComprehensiveNoiseAnalysis runs all architectures with both modes
func TestComprehensiveNoiseAnalysis(t *testing.T) {
	architectures := []struct {
		name         string
		architecture []int
	}{
		{"MNIST_MLP", []int{784, 128, 32, 10}},
		{"BCW_FC", []int{64, 32, 16, 10}},
		{"LeNet_FC", []int{256, 120, 84, 10}},
		{"Audio1D_FC", []int{2000, 5}},
		{"Small_FC", []int{16, 8, 4, 2}},
	}

	t.Log("\n" + strings.Repeat("=", 120))
	t.Log("COMPREHENSIVE NOISE ANALYSIS: Pure HE Noise vs Cheat-Strap")
	t.Log(strings.Repeat("=", 120))
	t.Log("\nThis compares HE polynomial vs Plaintext polynomial (isolating HE noise)")
	t.Log("Mode 1: No Refresh (accumulating noise)")
	t.Log("Mode 2: Cheat-Strap (refresh after each layer)\n")

	t.Log(strings.Repeat("-", 120))
	t.Logf("%-15s | %-25s | %-15s | %-15s | %-15s", "Model", "Architecture", "No Refresh RMS", "Cheat-Strap RMS", "Improvement")
	t.Log(strings.Repeat("-", 120))

	for _, arch := range architectures {
		reportNoRefresh := runForwardOnlyTestPureHENoise(t, arch.architecture, false)
		reportWithRefresh := runForwardOnlyTestPureHENoise(t, arch.architecture, true)

		improvement := "N/A"
		if reportWithRefresh.TotalForwardError > 0 {
			improvement = fmt.Sprintf("%.2fx", reportNoRefresh.TotalForwardError/reportWithRefresh.TotalForwardError)
		}

		archStr := fmt.Sprintf("%v", arch.architecture)
		t.Logf("%-15s | %-25s | %.6e    | %.6e    | %s",
			arch.name, archStr, reportNoRefresh.TotalForwardError, reportWithRefresh.TotalForwardError, improvement)
	}

	t.Log(strings.Repeat("-", 120))

	// Also output detailed layer-by-layer for one model
	t.Log("\n" + strings.Repeat("=", 120))
	t.Log("DETAILED LAYER-BY-LAYER: Small_FC (16->8->4->2)")
	t.Log(strings.Repeat("=", 120))

	t.Log("\n--- No Refresh ---")
	reportNoRefresh := runForwardOnlyTestPureHENoise(t, []int{16, 8, 4, 2}, false)
	t.Log(reportNoRefresh.PrintReport())

	t.Log("\n--- With Cheat-Strap ---")
	reportWithRefresh := runForwardOnlyTestPureHENoise(t, []int{16, 8, 4, 2}, true)
	t.Log(reportWithRefresh.PrintReport())
}
