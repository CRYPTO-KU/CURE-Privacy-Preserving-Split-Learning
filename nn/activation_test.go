package nn

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/tensor"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// computeReLU3Poly applies the ReLU3 polynomial manually for ground truth comparison
func computeReLU3Poly(vals []float64) []float64 {
	// ReLU3 polynomial: 0.3183099 + 0.5*x + 0.2122066*x^2 + 0*x^3
	c0, c1, c2 := 0.3183099, 0.5, 0.2122066
	result := make([]float64, len(vals))
	for i, x := range vals {
		result[i] = c0 + c1*x + c2*x*x
	}
	return result
}

func TestActivationHE(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	act, err := layers.NewActivation("ReLU3", true, he)
	if err != nil {
		t.Fatalf("failed to create Activation: %v", err)
	}

	// Input tensor in the approximation range
	inputData := []float64{-0.8, -0.2, 0.1, 0.9}
	input := tensor.New(4)
	copy(input.Data, inputData)

	// Ground truth: Apply the same ReLU3 polynomial that HE uses
	// NOTE: The plaintext layer's Forward() uses TRUE ReLU, not polynomial!
	// For fair comparison, we compute the polynomial manually
	expected := computeReLU3Poly(inputData)

	t.Logf("Input: %v", inputData)
	t.Logf("Expected polynomial output: %v", expected)

	// Encode and encrypt input
	pt := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	err = he.Encoder.Encode(input.Data, pt)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}
	ct, err := he.Encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("EncryptNew error: %v", err)
	}

	// Encrypted forward
	ctOut, err := act.ForwardCipher(ct)
	if err != nil {
		t.Fatalf("ForwardCipher failed: %v", err)
	}

	// Decrypt and decode
	ptOut := he.Decryptor.DecryptNew(ctOut)
	decoded := make([]complex128, he.Params.MaxSlots())
	err = he.Encoder.Decode(ptOut, decoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	t.Logf("HE polynomial output: %v", []float64{real(decoded[0]), real(decoded[1]), real(decoded[2]), real(decoded[3])})

	// Compare HE output to polynomial ground truth (should be very close, ~1e-8)
	for i := 0; i < 4; i++ {
		diff := math.Abs(real(decoded[i]) - expected[i])
		if diff > 1e-6 { // Tolerance for HE noise
			t.Errorf("mismatch at index %d: HE=%f, expected=%f, diff=%e",
				i, real(decoded[i]), expected[i], diff)
		} else {
			t.Logf("index %d: HE=%f, expected=%f, diff=%e ✓",
				i, real(decoded[i]), expected[i], diff)
		}
	}
}

func TestActivationHEBatchStats(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	act, err := layers.NewActivation("ReLU3", true, he)
	if err != nil {
		t.Fatalf("failed to create Activation: %v", err)
	}
	plainAct, _ := layers.NewActivation("ReLU3", false, nil)

	n := 100
	vecLen := 4
	divergences := make([]float64, 0, n*vecLen)
	for trial := 0; trial < n; trial++ {
		inputData := make([]float64, vecLen)
		for i := range inputData {
			inputData[i] = 2*rand.Float64() - 1 // Uniform in [-1,1]
		}
		input := tensor.New(vecLen)
		copy(input.Data, inputData)

		// Ground truth (plaintext)
		gt, err := plainAct.Forward(input)
		if err != nil {
			t.Fatalf("plaintext forward failed: %v", err)
		}
		gtTensor := gt.(*tensor.Tensor)

		// Encode and encrypt input
		pt := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
		err = he.Encoder.Encode(input.Data, pt)
		if err != nil {
			t.Fatalf("encode error: %v", err)
		}
		ct, err := he.Encryptor.EncryptNew(pt)
		if err != nil {
			t.Fatalf("EncryptNew error: %v", err)
		}

		// Encrypted forward
		ctOut, err := act.ForwardCipher(ct)
		if err != nil {
			t.Fatalf("ForwardCipher failed: %v", err)
		}

		// Decrypt and decode
		ptOut := he.Decryptor.DecryptNew(ctOut)
		decoded := make([]complex128, he.Params.MaxSlots())
		err = he.Encoder.Decode(ptOut, decoded)
		if err != nil {
			t.Fatalf("decode error: %v", err)
		}

		// Collect divergences
		for i := 0; i < vecLen; i++ {
			diff := math.Abs(real(decoded[i]) - gtTensor.Data[i])
			divergences = append(divergences, diff)
		}
	}
	// Compute statistics
	sum, min, max := 0.0, math.MaxFloat64, -math.MaxFloat64
	for _, d := range divergences {
		sum += d
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	mean := sum / float64(len(divergences))
	var sqsum float64
	for _, d := range divergences {
		sqsum += (d - mean) * (d - mean)
	}
	stddev := math.Sqrt(sqsum / float64(len(divergences)))
	fmt.Printf("[ActivationHEBatchStats] n=%d, mean=%.6g, stddev=%.6g, min=%.6g, max=%.6g\n", len(divergences), mean, stddev, min, max)
}

func TestActivationHEBackward(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	act, err := layers.NewActivation("ReLU3", true, he)
	if err != nil {
		t.Fatalf("failed to create Activation: %v", err)
	}
	plainDeriv, _ := layers.NewActivation("ReLU3_deriv", false, nil)

	inputData := []float64{-0.8, -0.2, 0.1, 0.9}
	input := tensor.New(len(inputData))
	copy(input.Data, inputData)

	// Encrypt input
	pt := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	he.Encoder.Encode(inputData, pt)
	ct, err := he.Encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("EncryptNew error: %v", err)
	}

	// Forward (HE)
	_, err = act.ForwardCipher(ct)
	if err != nil {
		t.Fatalf("ForwardCipher error: %v", err)
	}

	// Make a random gradOut
	gradData := make([]float64, len(inputData))
	for i := range gradData {
		gradData[i] = 2*rand.Float64() - 1
	}
	ptGrad := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	he.Encoder.Encode(gradData, ptGrad)
	ctGrad, err := he.Encryptor.EncryptNew(ptGrad)
	if err != nil {
		t.Fatalf("EncryptNew error: %v", err)
	}

	// Backward (HE)
	ctGradIn, err := act.BackwardHE(ctGrad)
	if err != nil {
		t.Fatalf("BackwardHE error: %v", err)
	}

	// Decrypt result
	ptGradIn := he.Decryptor.DecryptNew(ctGradIn)
	valsGradIn := make([]complex128, he.Params.MaxSlots())
	he.Encoder.Decode(ptGradIn, valsGradIn)

	// Shadow model: gradIn = gradOut * ReLU3'(input)
	plainDerivOutIface, _ := plainDeriv.Forward(input)
	plainDerivOut := plainDerivOutIface.(*tensor.Tensor)
	gradInShadow := tensor.New(len(inputData))
	for i := range gradInShadow.Data {
		gradInShadow.Data[i] = gradData[i] * plainDerivOut.Data[i]
	}

	// Print slotwise differences
	for i := range gradInShadow.Data {
		diff := math.Abs(real(valsGradIn[i]) - gradInShadow.Data[i])
		if diff > 1e-2 {
			t.Errorf("slot %d: got %v want %v diff %v", i, real(valsGradIn[i]), gradInShadow.Data[i], diff)
		} else {
			t.Logf("slot %d: got %v want %v diff %v", i, real(valsGradIn[i]), gradInShadow.Data[i], diff)
		}
	}
}
