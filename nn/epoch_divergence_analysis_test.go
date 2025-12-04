package nn

import (
	"math"
	"math/rand"
	"sort"
	"strings"
	"testing"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/tensor"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// EpochDivergenceStats stores statistics for epoch-level analysis
type EpochDivergenceStats struct {
	ModelName      string
	SplitIndex     int // l_n value (number of HE layers)
	NumTrials      int // Number of samples simulated
	EpochSize      int // Typical epoch size for this dataset
	MeanRMS        float64
	StdRMS         float64
	MinRMS         float64
	MaxRMS         float64
	Percentile99   float64 // 99th percentile of RMS errors
	ExpectedMaxRMS float64 // Expected max RMS for full epoch (using EVT)
	MeanMaxAbs     float64 // Mean of max absolute errors
	ExpectedMaxAbs float64 // Expected max absolute error in epoch
}

// runSingleTrialMLP runs one forward pass with random input and returns RMS and MaxAbs errors
func runSingleTrialMLP(heCtx *ckkswrapper.HeContext,
	heLinear1, heLinear2 *layers.Linear,
	plainLinear1, plainLinear2 *layers.Linear,
	heActivation *layers.Activation,
	inputDim, hidden1, outputDim int) (rmsErr, maxAbsErr float64) {

	slots := heCtx.Params.MaxSlots()

	// Random input (simulating one MNIST sample)
	input := make([]float64, inputDim)
	for i := range input {
		input[i] = rand.Float64() // Normalized pixel values [0,1]
	}

	// Encrypt input
	inputVec := make([]complex128, slots)
	for i := 0; i < inputDim; i++ {
		inputVec[i] = complex(input[i], 0)
	}
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec, pt)
	ct, _ := heCtx.Encryptor.EncryptNew(pt)

	// === Layer 1: Linear ===
	ctOut, _ := heLinear1.ForwardCipherMasked(ct)

	// Plain computation
	plainInput := tensor.New(inputDim)
	copy(plainInput.Data, input)
	plainOut, _ := plainLinear1.ForwardPlaintext(plainInput)

	// Cheat-strap (clean refresh)
	ctOut = cleanRefreshForEpoch(heCtx, ctOut, hidden1)

	// === Layer 2: ReLU3 Activation ===
	ctAct, _ := heActivation.ForwardCipher(ctOut)
	plainActVals := applyReLU3PolyEpoch(decryptToSliceEpoch(heCtx, ctOut, hidden1))

	// Cheat-strap
	ctAct = cleanRefreshForEpoch(heCtx, ctAct, hidden1)
	plainSynced := decryptToSliceEpoch(heCtx, ctAct, hidden1)

	// === Layer 3: Linear 2 (output layer) ===
	ctOut2, _ := heLinear2.ForwardCipherMasked(ctAct)

	// Plain computation for layer 2
	plainInput2 := tensor.New(hidden1)
	copy(plainInput2.Data, plainSynced)
	plainOut2, _ := plainLinear2.ForwardPlaintext(plainInput2)

	// Calculate final error
	heVals := decryptToSliceEpoch(heCtx, ctOut2, outputDim)
	rmsErr = calcRMSErrorEpoch(heVals, plainOut2.Data[:outputDim])
	maxAbsErr = calcMaxAbsErrorEpoch(heVals, plainOut2.Data[:outputDim])

	_ = plainOut // suppress unused
	_ = plainActVals

	return rmsErr, maxAbsErr
}

// Helper functions (duplicated to avoid import issues)
func cleanRefreshForEpoch(heCtx *ckkswrapper.HeContext, ct *rlwe.Ciphertext, validSize int) *rlwe.Ciphertext {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)

	for i := validSize; i < len(decoded); i++ {
		decoded[i] = 0
	}

	newPt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(decoded, newPt)
	newCt, _ := heCtx.Encryptor.EncryptNew(newPt)
	return newCt
}

func decryptToSliceEpoch(heCtx *ckkswrapper.HeContext, ct *rlwe.Ciphertext, numVals int) []float64 {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)
	result := make([]float64, numVals)
	for i := 0; i < numVals; i++ {
		result[i] = real(decoded[i])
	}
	return result
}

func applyReLU3PolyEpoch(vals []float64) []float64 {
	result := make([]float64, len(vals))
	for i, x := range vals {
		result[i] = 0.3183099 + 0.5*x + 0.2122066*x*x
	}
	return result
}

func calcRMSErrorEpoch(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(n))
}

func calcMaxAbsErrorEpoch(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	maxErr := 0.0
	for i := 0; i < n; i++ {
		diff := math.Abs(a[i] - b[i])
		if diff > maxErr {
			maxErr = diff
		}
	}
	return maxErr
}

// estimateExpectedMax uses order statistics to estimate expected max for N samples
// For N iid samples from a distribution, E[max] ≈ μ + σ * √(2*ln(N))  (for normal-ish distributions)
func estimateExpectedMax(mean, std float64, n int) float64 {
	if std == 0 {
		return mean
	}
	// Gumbel approximation for expected maximum
	return mean + std*math.Sqrt(2*math.Log(float64(n)))
}

// TestEpochDivergenceSimpleMLP simulates multiple samples for Simple MLP (784->128->10)
func TestEpochDivergenceSimpleMLP(t *testing.T) {
	rand.Seed(42)

	numTrials := 10    // Number of samples to simulate (reduced for speed)
	epochSize := 60000 // MNIST training set size

	t.Log("\n" + strings.Repeat("═", 100))
	t.Log("EPOCH DIVERGENCE ANALYSIS: Simple MLP (784 -> 128 -> 10)")
	t.Logf("Split Index l_n = 2 (Linear1 + ReLU3 under HE)")
	t.Logf("Running %d trials to estimate epoch-level divergence...", numTrials)
	t.Log(strings.Repeat("═", 100))

	heCtx := ckkswrapper.NewHeContext()

	inputDim := 784
	hidden1 := 128
	outputDim := 10

	// Create layers
	heLinear1 := layers.NewLinear(inputDim, hidden1, true, heCtx)
	heLinear2 := layers.NewLinear(hidden1, outputDim, true, heCtx)
	plainLinear1 := layers.NewLinear(inputDim, hidden1, false, nil)
	plainLinear2 := layers.NewLinear(hidden1, outputDim, false, nil)

	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Sync weights
	scale := math.Sqrt(2.0 / float64(inputDim+hidden1))
	for j := 0; j < hidden1; j++ {
		for k := 0; k < inputDim; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heLinear1.W.Data[j*inputDim+k] = w
			plainLinear1.W.Data[j*inputDim+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heLinear1.B.Data[j] = b
		plainLinear1.B.Data[j] = b
	}
	heLinear1.SyncHE()

	scale = math.Sqrt(2.0 / float64(hidden1+outputDim))
	for j := 0; j < outputDim; j++ {
		for k := 0; k < hidden1; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heLinear2.W.Data[j*hidden1+k] = w
			plainLinear2.W.Data[j*hidden1+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heLinear2.B.Data[j] = b
		plainLinear2.B.Data[j] = b
	}
	heLinear2.SyncHE()

	// Run trials
	rmsErrors := make([]float64, numTrials)
	maxAbsErrors := make([]float64, numTrials)

	for i := 0; i < numTrials; i++ {
		rms, maxAbs := runSingleTrialMLP(heCtx, heLinear1, heLinear2, plainLinear1, plainLinear2,
			heActivation, inputDim, hidden1, outputDim)
		rmsErrors[i] = rms
		maxAbsErrors[i] = maxAbs

		if (i+1)%20 == 0 {
			t.Logf("  Completed %d/%d trials...", i+1, numTrials)
		}
	}

	// Calculate statistics
	stats := calculateStats(rmsErrors, maxAbsErrors, epochSize)
	stats.ModelName = "Simple MLP"
	stats.SplitIndex = 2
	stats.NumTrials = numTrials
	stats.EpochSize = epochSize

	// Print results
	t.Log("\n" + strings.Repeat("-", 100))
	t.Log("DIVERGENCE STATISTICS (per sample):")
	t.Log(strings.Repeat("-", 100))
	t.Logf("  Mean RMS Error:      %.6e", stats.MeanRMS)
	t.Logf("  Std Dev RMS:         %.6e", stats.StdRMS)
	t.Logf("  Min RMS:             %.6e", stats.MinRMS)
	t.Logf("  Max RMS (observed):  %.6e", stats.MaxRMS)
	t.Logf("  99th Percentile:     %.6e", stats.Percentile99)
	t.Log("")
	t.Logf("  Mean Max Abs Error:  %.6e", stats.MeanMaxAbs)

	t.Log("\n" + strings.Repeat("-", 100))
	t.Logf("EPOCH-LEVEL ESTIMATES (N = %d samples):", epochSize)
	t.Log(strings.Repeat("-", 100))
	t.Logf("  Expected Max RMS in Epoch:     %.6e", stats.ExpectedMaxRMS)
	t.Logf("  Expected Max Abs Err in Epoch: %.6e", stats.ExpectedMaxAbs)
	t.Log(strings.Repeat("═", 100))
}

// TestEpochDivergenceMLP tests the larger MLP (784->128->64->10)
func TestEpochDivergenceMLP(t *testing.T) {
	rand.Seed(42)

	numTrials := 50
	epochSize := 60000

	t.Log("\n" + strings.Repeat("═", 100))
	t.Log("EPOCH DIVERGENCE ANALYSIS: MLP (784 -> 128 -> 64 -> 10)")
	t.Logf("Split Index l_n = 2 (Linear1 + ReLU3 under HE)")
	t.Logf("Running %d trials...", numTrials)
	t.Log(strings.Repeat("═", 100))

	heCtx := ckkswrapper.NewHeContext()

	inputDim := 784
	hidden1 := 128
	hidden2 := 64
	outputDim := 10

	// Create layers
	heLinear1 := layers.NewLinear(inputDim, hidden1, true, heCtx)
	plainLinear1 := layers.NewLinear(inputDim, hidden1, false, nil)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Sync weights
	scale := math.Sqrt(2.0 / float64(inputDim+hidden1))
	for j := 0; j < hidden1; j++ {
		for k := 0; k < inputDim; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			heLinear1.W.Data[j*inputDim+k] = w
			plainLinear1.W.Data[j*inputDim+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		heLinear1.B.Data[j] = b
		plainLinear1.B.Data[j] = b
	}
	heLinear1.SyncHE()

	slots := heCtx.Params.MaxSlots()
	rmsErrors := make([]float64, numTrials)
	maxAbsErrors := make([]float64, numTrials)

	for trial := 0; trial < numTrials; trial++ {
		// Random input
		input := make([]float64, inputDim)
		for i := range input {
			input[i] = rand.Float64()
		}

		// Encrypt
		inputVec := make([]complex128, slots)
		for i := 0; i < inputDim; i++ {
			inputVec[i] = complex(input[i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)

		// Linear1
		ctOut, _ := heLinear1.ForwardCipherMasked(ct)
		plainInput := tensor.New(inputDim)
		copy(plainInput.Data, input)
		plainOut, _ := plainLinear1.ForwardPlaintext(plainInput)

		// Cheat-strap
		ctOut = cleanRefreshForEpoch(heCtx, ctOut, hidden1)

		// ReLU3
		ctAct, _ := heActivation.ForwardCipher(ctOut)
		plainActVals := applyReLU3PolyEpoch(decryptToSliceEpoch(heCtx, ctOut, hidden1))

		// Final error after 2 HE layers
		heVals := decryptToSliceEpoch(heCtx, ctAct, hidden1)
		rmsErrors[trial] = calcRMSErrorEpoch(heVals, plainActVals)
		maxAbsErrors[trial] = calcMaxAbsErrorEpoch(heVals, plainActVals)

		_ = plainOut
		_ = hidden2
		_ = outputDim
	}

	stats := calculateStats(rmsErrors, maxAbsErrors, epochSize)

	t.Log("\n" + strings.Repeat("-", 100))
	t.Log("DIVERGENCE STATISTICS (per sample, after l_n=2 layers):")
	t.Log(strings.Repeat("-", 100))
	t.Logf("  Mean RMS Error:      %.6e", stats.MeanRMS)
	t.Logf("  Std Dev RMS:         %.6e", stats.StdRMS)
	t.Logf("  Max RMS (observed):  %.6e", stats.MaxRMS)
	t.Logf("  99th Percentile:     %.6e", stats.Percentile99)
	t.Log("")
	t.Logf("EPOCH-LEVEL (N = %d):", epochSize)
	t.Logf("  Expected Max RMS:    %.6e", stats.ExpectedMaxRMS)
	t.Logf("  Expected Max Abs:    %.6e", stats.ExpectedMaxAbs)
	t.Log(strings.Repeat("═", 100))
}

// TestEpochDivergenceLeNet tests LeNet with l_n=2 (Conv1 + ReLU3)
func TestEpochDivergenceLeNet(t *testing.T) {
	rand.Seed(42)

	numTrials := 30 // Fewer trials since conv is slower
	epochSize := 60000

	t.Log("\n" + strings.Repeat("═", 100))
	t.Log("EPOCH DIVERGENCE ANALYSIS: LeNet")
	t.Logf("Split Index l_n = 2 (Conv1 + ReLU3 under HE)")
	t.Logf("Running %d trials...", numTrials)
	t.Log(strings.Repeat("═", 100))

	heCtx := ckkswrapper.NewHeContext()

	inH, inW := 16, 16
	conv1InChan, conv1OutChan := 1, 6
	kernelSize := 5
	outH := inH - kernelSize + 1 // 12
	outW := inW - kernelSize + 1 // 12

	heConv1 := layers.NewConv2D(conv1InChan, conv1OutChan, kernelSize, kernelSize, true, heCtx)
	plainConv1 := layers.NewConv2D(conv1InChan, conv1OutChan, kernelSize, kernelSize, false, nil)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Set dimensions and sync HE
	heConv1.SetDimensions(inH, inW)
	plainConv1.SetDimensions(inH, inW)

	// Sync weights
	scale := math.Sqrt(2.0 / float64(conv1InChan*kernelSize*kernelSize))
	for oc := 0; oc < conv1OutChan; oc++ {
		for ic := 0; ic < conv1InChan; ic++ {
			for kh := 0; kh < kernelSize; kh++ {
				for kw := 0; kw < kernelSize; kw++ {
					idx := oc*conv1InChan*kernelSize*kernelSize + ic*kernelSize*kernelSize + kh*kernelSize + kw
					w := (rand.Float64() - 0.5) * 2 * scale
					heConv1.W.Data[idx] = w
					plainConv1.W.Data[idx] = w
				}
			}
		}
		heConv1.B.Data[oc] = (rand.Float64() - 0.5) * 0.1
		plainConv1.B.Data[oc] = heConv1.B.Data[oc]
	}
	heConv1.SyncHE()

	slots := heCtx.Params.MaxSlots()
	rmsErrors := make([]float64, numTrials)
	maxAbsErrors := make([]float64, numTrials)

	for trial := 0; trial < numTrials; trial++ {
		// Random image
		inputTensor := tensor.New(conv1InChan, inH, inW)
		for i := range inputTensor.Data {
			inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
		}

		// Encrypt
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

		// Conv1
		heConvOut, _ := heConv1.ForwardHE(inputCTs)
		plainConvOut, _ := plainConv1.ForwardPlain(inputTensor)

		// Decrypt conv output at proper positions
		heConvVals := decryptConv2DOutputEpoch(heCtx, heConvOut, outH, outW, inW)

		// Cheat-strap each channel
		for c := range heConvOut {
			heConvOut[c] = cleanRefreshConv2DEpoch(heCtx, heConvOut[c], outH, outW, inW)
		}
		plainConvSynced := decryptConv2DOutputEpoch(heCtx, heConvOut, outH, outW, inW)

		// ReLU3
		heActOut := make([]*rlwe.Ciphertext, len(heConvOut))
		for c := 0; c < len(heConvOut); c++ {
			heActOut[c], _ = heActivation.ForwardCipher(heConvOut[c])
		}
		plainActVals := applyReLU3PolyEpoch(plainConvSynced)

		heActVals := decryptConv2DOutputEpoch(heCtx, heActOut, outH, outW, inW)
		rmsErrors[trial] = calcRMSErrorEpoch(heActVals, plainActVals)
		maxAbsErrors[trial] = calcMaxAbsErrorEpoch(heActVals, plainActVals)

		_ = heConvVals
		_ = plainConvOut

		if (trial+1)%10 == 0 {
			t.Logf("  Completed %d/%d trials...", trial+1, numTrials)
		}
	}

	stats := calculateStats(rmsErrors, maxAbsErrors, epochSize)

	t.Log("\n" + strings.Repeat("-", 100))
	t.Log("DIVERGENCE STATISTICS (per sample, after l_n=2 layers):")
	t.Log(strings.Repeat("-", 100))
	t.Logf("  Mean RMS Error:      %.6e", stats.MeanRMS)
	t.Logf("  Std Dev RMS:         %.6e", stats.StdRMS)
	t.Logf("  Max RMS (observed):  %.6e", stats.MaxRMS)
	t.Logf("  99th Percentile:     %.6e", stats.Percentile99)
	t.Log("")
	t.Logf("EPOCH-LEVEL (N = %d):", epochSize)
	t.Logf("  Expected Max RMS:    %.6e", stats.ExpectedMaxRMS)
	t.Logf("  Expected Max Abs:    %.6e", stats.ExpectedMaxAbs)
	t.Log(strings.Repeat("═", 100))
}

// Helper for Conv2D output decryption
func decryptConv2DOutputEpoch(heCtx *ckkswrapper.HeContext, cts []*rlwe.Ciphertext, outH, outW, inW int) []float64 {
	numOutputs := len(cts) * outH * outW
	result := make([]float64, numOutputs)
	idx := 0
	for _, ct := range cts {
		pt := heCtx.Decryptor.DecryptNew(ct)
		decoded := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(pt, decoded)
		for oy := 0; oy < outH; oy++ {
			for ox := 0; ox < outW; ox++ {
				pos := oy*inW + ox
				result[idx] = real(decoded[pos])
				idx++
			}
		}
	}
	return result
}

func cleanRefreshConv2DEpoch(heCtx *ckkswrapper.HeContext, ct *rlwe.Ciphertext, outH, outW, inW int) *rlwe.Ciphertext {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)

	cleaned := make([]complex128, heCtx.Params.MaxSlots())
	for oy := 0; oy < outH; oy++ {
		for ox := 0; ox < outW; ox++ {
			pos := oy*inW + ox
			cleaned[pos] = decoded[pos]
		}
	}

	newPt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(cleaned, newPt)
	newCt, _ := heCtx.Encryptor.EncryptNew(newPt)
	return newCt
}

// calculateStats computes all statistics from trial data
func calculateStats(rmsErrors, maxAbsErrors []float64, epochSize int) EpochDivergenceStats {
	n := len(rmsErrors)

	// Sort for percentiles
	sortedRMS := make([]float64, n)
	copy(sortedRMS, rmsErrors)
	sort.Float64s(sortedRMS)

	sortedMaxAbs := make([]float64, n)
	copy(sortedMaxAbs, maxAbsErrors)
	sort.Float64s(sortedMaxAbs)

	// Mean and std
	var sumRMS, sumMaxAbs float64
	for i := 0; i < n; i++ {
		sumRMS += rmsErrors[i]
		sumMaxAbs += maxAbsErrors[i]
	}
	meanRMS := sumRMS / float64(n)
	meanMaxAbs := sumMaxAbs / float64(n)

	var varRMS, varMaxAbs float64
	for i := 0; i < n; i++ {
		varRMS += (rmsErrors[i] - meanRMS) * (rmsErrors[i] - meanRMS)
		varMaxAbs += (maxAbsErrors[i] - meanMaxAbs) * (maxAbsErrors[i] - meanMaxAbs)
	}
	stdRMS := math.Sqrt(varRMS / float64(n))
	stdMaxAbs := math.Sqrt(varMaxAbs / float64(n))

	// Percentile index
	p99Idx := int(0.99 * float64(n))
	if p99Idx >= n {
		p99Idx = n - 1
	}

	return EpochDivergenceStats{
		MeanRMS:        meanRMS,
		StdRMS:         stdRMS,
		MinRMS:         sortedRMS[0],
		MaxRMS:         sortedRMS[n-1],
		Percentile99:   sortedRMS[p99Idx],
		ExpectedMaxRMS: estimateExpectedMax(meanRMS, stdRMS, epochSize),
		MeanMaxAbs:     meanMaxAbs,
		ExpectedMaxAbs: estimateExpectedMax(meanMaxAbs, stdMaxAbs, epochSize),
		EpochSize:      epochSize,
	}
}

// TestEpochDivergenceSummaryTable generates the complete table data
func TestEpochDivergenceSummaryTable(t *testing.T) {
	t.Log("\n")
	t.Log(strings.Repeat("═", 140))
	t.Log("EPOCH-LEVEL DIVERGENCE SUMMARY TABLE")
	t.Log("All models with split index l_n = 2")
	t.Log(strings.Repeat("═", 140))

	// Table header
	t.Log("")
	t.Logf("%-20s | %-12s | %-12s | %-12s | %-12s | %-15s | %-15s",
		"Model", "Mean RMS", "Std RMS", "99th %ile", "Max Obs", "Exp Max (Epoch)", "Exp Max Abs")
	t.Log(strings.Repeat("-", 140))

	// Run subtests and collect data
	t.Run("SimpleMLP", TestEpochDivergenceSimpleMLP)
	t.Run("MLP", TestEpochDivergenceMLP)
	t.Run("LeNet", TestEpochDivergenceLeNet)
}

// TestGenerateTableData runs analysis and outputs data suitable for the paper table
func TestGenerateTableData(t *testing.T) {
	rand.Seed(42)

	t.Log("\n" + strings.Repeat("═", 120))
	t.Log("TABLE DATA FOR PAPER: HE Divergence per Epoch")
	t.Log("Mode: Cheat-strap enabled, l_n = 2")
	t.Log(strings.Repeat("═", 120))

	// Model configurations
	configs := []struct {
		name      string
		epochSize int
		testFunc  func(*testing.T, int) EpochDivergenceStats
	}{
		{"Simple MLP", 60000, runSimpleMLPTrials},
		{"MLP", 60000, runMLPTrials},
		{"LeNet", 60000, runLeNetTrials},
	}

	results := make([]EpochDivergenceStats, len(configs))

	for i, cfg := range configs {
		t.Logf("\nRunning %s trials...", cfg.name)
		results[i] = cfg.testFunc(t, 50)
		results[i].ModelName = cfg.name
		results[i].EpochSize = cfg.epochSize
	}

	// Output table
	t.Log("\n" + strings.Repeat("═", 120))
	t.Log("FINAL TABLE DATA:")
	t.Log(strings.Repeat("-", 120))
	t.Logf("%-15s | %-12s | %-12s | %-15s | %-18s",
		"Model", "Mean RMS", "Max RMS", "99th Percentile", "Expected Max/Epoch")
	t.Log(strings.Repeat("-", 120))

	for _, r := range results {
		t.Logf("%-15s | %.4e | %.4e | %.4e    | %.4e",
			r.ModelName, r.MeanRMS, r.MaxRMS, r.Percentile99, r.ExpectedMaxRMS)
	}
	t.Log(strings.Repeat("═", 120))

	// LaTeX table suggestion
	t.Log("\n" + strings.Repeat("═", 120))
	t.Log("SUGGESTED LaTeX COLUMN TO ADD:")
	t.Log(strings.Repeat("-", 120))
	t.Log(`
Add column: "E[Max RMS]/Epoch" or "ε_max" 

This represents the expected maximum per-sample RMS divergence 
that will occur during one epoch of training/inference.

Formula: E[max] ≈ μ + σ × √(2 × ln(N))
where N = epoch size (e.g., 60000 for MNIST)
`)
	t.Log(strings.Repeat("═", 120))
}

// Helper functions to run trials and return stats
func runSimpleMLPTrials(t *testing.T, numTrials int) EpochDivergenceStats {
	heCtx := ckkswrapper.NewHeContext()

	inputDim, hidden1, outputDim := 784, 128, 10

	heLinear1 := layers.NewLinear(inputDim, hidden1, true, heCtx)
	heLinear2 := layers.NewLinear(hidden1, outputDim, true, heCtx)
	plainLinear1 := layers.NewLinear(inputDim, hidden1, false, nil)
	plainLinear2 := layers.NewLinear(hidden1, outputDim, false, nil)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Sync weights
	syncLinearWeights(heLinear1, plainLinear1, inputDim, hidden1)
	syncLinearWeights(heLinear2, plainLinear2, hidden1, outputDim)

	rmsErrors := make([]float64, numTrials)
	maxAbsErrors := make([]float64, numTrials)

	for i := 0; i < numTrials; i++ {
		rms, maxAbs := runSingleTrialMLP(heCtx, heLinear1, heLinear2, plainLinear1, plainLinear2,
			heActivation, inputDim, hidden1, outputDim)
		rmsErrors[i] = rms
		maxAbsErrors[i] = maxAbs
	}

	return calculateStats(rmsErrors, maxAbsErrors, 60000)
}

func runMLPTrials(t *testing.T, numTrials int) EpochDivergenceStats {
	heCtx := ckkswrapper.NewHeContext()

	inputDim, hidden1 := 784, 128

	heLinear1 := layers.NewLinear(inputDim, hidden1, true, heCtx)
	plainLinear1 := layers.NewLinear(inputDim, hidden1, false, nil)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	syncLinearWeights(heLinear1, plainLinear1, inputDim, hidden1)

	slots := heCtx.Params.MaxSlots()
	rmsErrors := make([]float64, numTrials)
	maxAbsErrors := make([]float64, numTrials)

	for trial := 0; trial < numTrials; trial++ {
		input := make([]float64, inputDim)
		for i := range input {
			input[i] = rand.Float64()
		}

		inputVec := make([]complex128, slots)
		for i := 0; i < inputDim; i++ {
			inputVec[i] = complex(input[i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		heCtx.Encoder.Encode(inputVec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)

		ctOut, _ := heLinear1.ForwardCipherMasked(ct)
		ctOut = cleanRefreshForEpoch(heCtx, ctOut, hidden1)

		ctAct, _ := heActivation.ForwardCipher(ctOut)
		plainActVals := applyReLU3PolyEpoch(decryptToSliceEpoch(heCtx, ctOut, hidden1))

		heVals := decryptToSliceEpoch(heCtx, ctAct, hidden1)
		rmsErrors[trial] = calcRMSErrorEpoch(heVals, plainActVals)
		maxAbsErrors[trial] = calcMaxAbsErrorEpoch(heVals, plainActVals)
	}

	return calculateStats(rmsErrors, maxAbsErrors, 60000)
}

func runLeNetTrials(t *testing.T, numTrials int) EpochDivergenceStats {
	heCtx := ckkswrapper.NewHeContext()

	inH, inW := 16, 16
	conv1InChan, conv1OutChan := 1, 6
	kernelSize := 5
	outH := inH - kernelSize + 1
	outW := inW - kernelSize + 1

	heConv1 := layers.NewConv2D(conv1InChan, conv1OutChan, kernelSize, kernelSize, true, heCtx)
	plainConv1 := layers.NewConv2D(conv1InChan, conv1OutChan, kernelSize, kernelSize, false, nil)
	heActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	heConv1.SetDimensions(inH, inW)
	plainConv1.SetDimensions(inH, inW)
	syncConvWeights(heConv1, plainConv1, conv1OutChan, conv1InChan, kernelSize)

	slots := heCtx.Params.MaxSlots()
	rmsErrors := make([]float64, numTrials)
	maxAbsErrors := make([]float64, numTrials)

	for trial := 0; trial < numTrials; trial++ {
		inputTensor := tensor.New(conv1InChan, inH, inW)
		for i := range inputTensor.Data {
			inputTensor.Data[i] = (rand.Float64() - 0.5) * 2.0
		}

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

		heConvOut, _ := heConv1.ForwardHE(inputCTs)

		for c := range heConvOut {
			heConvOut[c] = cleanRefreshConv2DEpoch(heCtx, heConvOut[c], outH, outW, inW)
		}
		plainConvSynced := decryptConv2DOutputEpoch(heCtx, heConvOut, outH, outW, inW)

		heActOut := make([]*rlwe.Ciphertext, len(heConvOut))
		for c := 0; c < len(heConvOut); c++ {
			heActOut[c], _ = heActivation.ForwardCipher(heConvOut[c])
		}
		plainActVals := applyReLU3PolyEpoch(plainConvSynced)

		heActVals := decryptConv2DOutputEpoch(heCtx, heActOut, outH, outW, inW)
		rmsErrors[trial] = calcRMSErrorEpoch(heActVals, plainActVals)
		maxAbsErrors[trial] = calcMaxAbsErrorEpoch(heActVals, plainActVals)
	}

	return calculateStats(rmsErrors, maxAbsErrors, 60000)
}

func syncLinearWeights(he, plain *layers.Linear, inDim, outDim int) {
	scale := math.Sqrt(2.0 / float64(inDim+outDim))
	for j := 0; j < outDim; j++ {
		for k := 0; k < inDim; k++ {
			w := (rand.Float64() - 0.5) * 2 * scale
			he.W.Data[j*inDim+k] = w
			plain.W.Data[j*inDim+k] = w
		}
		b := (rand.Float64() - 0.5) * 0.1
		he.B.Data[j] = b
		plain.B.Data[j] = b
	}
	he.SyncHE()
}

func syncConvWeights(he, plain *layers.Conv2D, outChan, inChan, kernelSize int) {
	scale := math.Sqrt(2.0 / float64(inChan*kernelSize*kernelSize))
	for oc := 0; oc < outChan; oc++ {
		for ic := 0; ic < inChan; ic++ {
			for kh := 0; kh < kernelSize; kh++ {
				for kw := 0; kw < kernelSize; kw++ {
					idx := oc*inChan*kernelSize*kernelSize + ic*kernelSize*kernelSize + kh*kernelSize + kw
					w := (rand.Float64() - 0.5) * 2 * scale
					he.W.Data[idx] = w
					plain.W.Data[idx] = w
				}
			}
		}
		he.B.Data[oc] = (rand.Float64() - 0.5) * 0.1
		plain.B.Data[oc] = he.B.Data[oc]
	}
	he.SyncHE()
}
