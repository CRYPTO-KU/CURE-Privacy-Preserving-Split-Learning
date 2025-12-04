//go:build !exclude_he
// +build !exclude_he

package nn

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/tensor"
	"fmt"
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestLinearForwardCipherMasked(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	inDim := 4
	outDim := 4

	// Example weights and bias
	W := [][]float64{
		{1, 2, 3, 4},
		{2, 0, 1, 1},
		{0, 1, 0, 1},
		{1, 1, 1, 1},
	}
	b := []float64{10, 20, 30, 40}

	// Example input
	x := []float64{1, 2, 3, 4}

	// Compute expected output
	want := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			want[j] += W[j][i] * x[i]
		}
		want[j] += b[j]
	}

	// Create Linear layer
	lin := layers.NewLinear(inDim, outDim, true, he)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			lin.W.Data[j*inDim+i] = W[j][i]
		}
		lin.B.Data[j] = b[j]
	}
	lin.SyncHE()

	// Encode and encrypt input
	slots := he.Params.MaxSlots()
	xvec := make([]complex128, slots)
	for i := 0; i < inDim; i++ {
		xvec[i] = complex(x[i], 0)
	}
	// Debug: print type and values before encoding
	if inDim <= 8 {
		fmt.Printf("[Test] Input vector (before encode): %T %v\n", xvec, xvec[:inDim])
	}
	ptX := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	ptX.Scale = he.Params.DefaultScale()
	if err := he.Encoder.Encode(xvec, ptX); err != nil {
		t.Fatalf("encode error: %v", err)
	}
	// Debug: decode immediately after encoding
	decodedInput := make([]complex128, slots)
	he.Encoder.Decode(ptX, decodedInput)
	if inDim <= 8 {
		fmt.Printf("[Test] Input vector (after encode/decode): %v\n", decodedInput[:inDim])
	}
	ctX, _ := he.Encryptor.EncryptNew(ptX)

	// Minimal roundtrip for input
	ctX_round, err := he.Encryptor.EncryptNew(ptX)
	if err != nil {
		t.Fatalf("encrypt error: %v", err)
	}
	ptX_round := he.Decryptor.DecryptNew(ctX_round)
	decodedInputRound := make([]complex128, slots)
	he.Encoder.Decode(ptX_round, decodedInputRound)
	if inDim <= 8 {
		fmt.Printf("[Test] Input vector (after encrypt/decrypt/decode): %v\n", decodedInputRound[:inDim])
	}

	// Minimal roundtrip for weights
	for j := 0; j < outDim; j++ {
		wrow := make([]complex128, slots)
		for i := 0; i < inDim; i++ {
			wrow[i] = complex(W[j][i], 0)
		}
		ptW := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
		ptW.Scale = he.Params.DefaultScale()
		if err := he.Encoder.Encode(wrow, ptW); err != nil {
			t.Fatalf("weight encode error: %v", err)
		}
		ctW, err := he.Encryptor.EncryptNew(ptW)
		if err != nil {
			t.Fatalf("weight encrypt error: %v", err)
		}
		ptW_round := he.Decryptor.DecryptNew(ctW)
		decodedW := make([]complex128, slots)
		he.Encoder.Decode(ptW_round, decodedW)
		if inDim <= 8 {
			fmt.Printf("[Test] Weight row %d (after encrypt/decrypt/decode): %v\n", j, decodedW[:inDim])
		}
	}

	// Call ForwardCipherMasked
	ctY, err := lin.ForwardCipherMasked(ctX)
	if err != nil {
		t.Fatalf("ForwardCipherMasked error: %v", err)
	}

	// Decrypt and check output
	decY := he.Decryptor.DecryptNew(ctY)
	valsY := make([]complex128, slots)
	he.Encoder.Decode(decY, valsY)
	for j := 0; j < outDim; j++ {
		got := real(valsY[j])
		if math.Abs(got-want[j]) > 1e-2 {
			t.Errorf("slot %d: got %.4f want %.4f", j, got, want[j])
		} else {
			t.Logf("slot %d OK: got %.4f want %.4f", j, got, want[j])
		}
	}
}

func TestLinearForwardCipherMasked784x128(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	inDim := 784
	outDim := 128

	// Example weights and bias (patterned for reproducibility)
	W := make([][]float64, outDim)
	b := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		W[j] = make([]float64, inDim)
		for i := 0; i < inDim; i++ {
			W[j][i] = float64((j+1)*(i+1)%10 + 1) // simple pattern
		}
		b[j] = float64(j % 5)
	}

	// Example input
	x := make([]float64, inDim)
	for i := 0; i < inDim; i++ {
		x[i] = float64(i%7 + 1)
	}

	// Compute expected output for first 4 slots
	want := make([]float64, 4)
	for j := 0; j < 4; j++ {
		for i := 0; i < inDim; i++ {
			want[j] += W[j][i] * x[i]
		}
		want[j] += b[j]
	}

	// Create Linear layer
	lin := layers.NewLinear(inDim, outDim, true, he)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			lin.W.Data[j*inDim+i] = W[j][i]
		}
		lin.B.Data[j] = b[j]
	}
	lin.SyncHE()

	// Encrypt input
	slots := he.Params.MaxSlots()
	xvec := make([]complex128, slots)
	for i := 0; i < inDim; i++ {
		xvec[i] = complex(x[i], 0)
	}
	ptX := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	he.Encoder.Encode(xvec, ptX)
	ctX, _ := he.Encryptor.EncryptNew(ptX)

	// Forward
	ctOut, err := lin.ForwardCipherMasked(ctX)
	if err != nil {
		t.Fatalf("ForwardCipherMasked error: %v", err)
	}

	// Decrypt and check first 4 slots
	dec := he.Decryptor.DecryptNew(ctOut)
	vals := make([]complex128, slots)
	he.Encoder.Decode(dec, vals)
	for j := 0; j < 4; j++ {
		got := real(vals[j])
		if !almostEqual(got, want[j], 1e-2) {
			t.Errorf("slot %d: got %.4f want %.4f", j, got, want[j])
		} else {
			t.Logf("slot %d OK: got %.4f want %.4f", j, got, want[j])
		}
	}
}

func TestLinearPlaintextAndHEAgreement(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	inDim := 8
	outDim := 4

	// Deterministic weights and bias
	lin := layers.NewLinear(inDim, outDim, false, nil)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			lin.W.Data[j*inDim+i] = float64(i+1) * float64(j+1) / 10.0
		}
		lin.B.Data[j] = float64(j) / 5.0
	}

	// Input vector
	xData := make([]float64, inDim)
	for i := 0; i < inDim; i++ {
		xData[i] = float64(i+1) / 2.0
	}
	x := &tensor.Tensor{Data: xData, Shape: []int{inDim}}

	// Plaintext forward
	yPlain, err := lin.ForwardPlaintext(x)
	if err != nil {
		t.Fatalf("ForwardPlaintext error: %v", err)
	}

	// Switch to encrypted mode and sync HE weights
	linEncrypted := layers.NewLinear(inDim, outDim, true, he)
	// Copy weights from plaintext layer
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			linEncrypted.W.Data[j*inDim+i] = lin.W.Data[j*inDim+i]
		}
		linEncrypted.B.Data[j] = lin.B.Data[j]
	}
	linEncrypted.SyncHE()

	// Test that encrypted layer has HE context
	if !linEncrypted.Encrypted() {
		t.Fatal("encrypted layer should have Encrypted() == true")
	}

	// Encode and encrypt input
	slots := he.Params.MaxSlots()
	xvec := make([]complex128, slots)
	for i := 0; i < inDim; i++ {
		xvec[i] = complex(xData[i], 0)
	}
	ptX := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	he.Encoder.Encode(xvec, ptX)
	ctX, _ := he.Encryptor.EncryptNew(ptX)

	// HE forward
	ctY, err := linEncrypted.ForwardCipherMasked(ctX)
	if err != nil {
		t.Fatalf("ForwardCipherMasked error: %v", err)
	}
	decY := he.Decryptor.DecryptNew(ctY)
	valsY := make([]complex128, slots)
	he.Encoder.Decode(decY, valsY)

	eps := 1e-3
	for j := 0; j < outDim; j++ {
		got := real(valsY[j])
		want := yPlain.Data[j]
		if math.Abs(got-want) > eps {
			t.Errorf("slot %d: got %.6f want %.6f", j, got, want)
		} else {
			t.Logf("slot %d OK: got %.6f want %.6f", j, got, want)
		}
	}
}

func TestLinearForwardBatchedInput(t *testing.T) {
	heCtx := ckkswrapper.NewHeContext()
	layer := layers.NewLinear(784, 128, true, heCtx)
	layer.SyncHE()
	inputVec := tensor.New(784)
	for i := range inputVec.Data {
		inputVec.Data[i] = 1.0
	} // Use all ones

	// Print input vector (first 10 values)
	t.Logf("Input vector (first 10): %v", inputVec.Data[:10])

	ptInput := rlwe.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec.Data, ptInput)

	// Print encoded plaintext (first 10 slots)
	decoded := make([]complex128, 784)
	heCtx.Encoder.Decode(ptInput, decoded)
	t.Logf("Encoded plaintext (first 10): %v", decoded[:10])

	// Print weights (first 10 values of first output neuron)
	if layer.W != nil && len(layer.W.Data) >= 10 {
		t.Logf("Weights (first 10): %v", layer.W.Data[:10])
	}

	ctInput, err := heCtx.Encryptor.EncryptNew(ptInput)
	if err != nil {
		t.Fatalf("Encrypt input: %v", err)
	}

	// Print IsBatched status if possible (not directly available, but we can infer from encoding)
	t.Logf("Plaintext decoded length: %d", len(decoded))

	_, err = layer.ForwardCipherMasked(ctInput)
	if err != nil {
		t.Fatalf("Linear ForwardCipherMasked: %v", err)
	}
}

func TestCKKSEncodeDecodeSanity(t *testing.T) {
	heCtx := ckkswrapper.NewHeContext()
	slots := heCtx.Params.MaxSlots()
	vec := make([]float64, slots)
	for i := 0; i < 8; i++ {
		vec[i] = float64(i + 1)
	}
	pt := rlwe.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	pt.Scale = heCtx.Params.DefaultScale() // Ensure correct scale

	if err := heCtx.Encoder.Encode(vec, pt); err != nil {
		t.Fatalf("encode error: %v", err)
	}
	decoded := make([]complex128, slots)
	if err := heCtx.Encoder.Decode(pt, decoded); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	t.Logf("Sanity check (float64): Encoded-then-decoded (first 8): %v", decoded[:8])

	// Full roundtrip: encode -> encrypt -> decrypt -> decode
	ct, err := heCtx.Encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt error: %v", err)
	}
	pt2 := heCtx.Decryptor.DecryptNew(ct)
	decoded2 := make([]complex128, slots)
	if err := heCtx.Encoder.Decode(pt2, decoded2); err != nil {
		t.Fatalf("decode after decrypt error: %v", err)
	}
	// Check correctness: first 8 values should be close to input
	for i := 0; i < 8; i++ {
		if math.Abs(real(decoded2[i])-vec[i]) > 1e-3 {
			t.Errorf("slot %d: got %.8f, want %.8f", i, real(decoded2[i]), vec[i])
		}
	}
	t.Logf("Sanity check (float64): Encoded->Encrypted->Decrypted->Decoded (first 8): %v", decoded2[:8])
}

// almostEqual checks if two floats are within tol
func almostEqual(a, b, eps float64) bool {
	if a > b {
		return a-b < eps
	}
	return b-a < eps
}

// TestLinearForwardCipherMaskedShadow tests the Linear HE forward pass with a shadow plaintext model for correctness.
func TestLinearForwardCipherMaskedShadow(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	inDim := 4
	outDim := 4

	// Example weights and bias
	W := [][]float64{
		{1, 2, 3, 4},
		{2, 0, 1, 1},
		{0, 1, 0, 1},
		{1, 1, 1, 1},
	}
	b := []float64{10, 20, 30, 40}

	// Example input
	x := []float64{1, 2, 3, 4}

	// Create Linear layer (HE)
	lin := layers.NewLinear(inDim, outDim, true, he)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			lin.W.Data[j*inDim+i] = W[j][i]
		}
		lin.B.Data[j] = b[j]
	}
	lin.SyncHE()

	// Encode and encrypt input
	slots := he.Params.MaxSlots()
	xvec := make([]complex128, slots)
	for i := 0; i < inDim; i++ {
		xvec[i] = complex(x[i], 0)
	}
	ptX := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	ptX.Scale = he.Params.DefaultScale()
	if err := he.Encoder.Encode(xvec, ptX); err != nil {
		t.Fatalf("encode error: %v", err)
	}
	ctX, err := he.Encryptor.EncryptNew(ptX)
	if err != nil {
		t.Fatalf("encrypt error: %v", err)
	}

	// Run HE forward
	ctY, err := lin.ForwardCipherMasked(ctX)
	if err != nil {
		t.Fatalf("ForwardCipherMasked error: %v", err)
	}

	// Decrypt and check output
	decY := he.Decryptor.DecryptNew(ctY)
	valsY := make([]complex128, slots)
	if err := he.Encoder.Decode(decY, valsY); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	fmt.Printf("[Test] Decoded output vector (first 8 slots): %v\n", valsY[:8])

	// Compute shadow output (plaintext reference)
	shadow := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			shadow[j] += W[j][i] * x[i]
		}
		shadow[j] += b[j]
	}

	// Compare outputs
	for j := 0; j < outDim; j++ {
		got := real(valsY[j])
		want := shadow[j]
		fmt.Printf("[Test] Comparing slot %d: got %.4f, want %.4f\n", j, got, want)
		if math.Abs(got-want) > 1e-2 {
			t.Errorf("slot %d: got %.4f want %.4f", j, got, want)
		} else {
			t.Logf("slot %d OK: got %.4f want %.4f", j, got, want)
		}
	}
}

func TestLinearActivationHEWithShadow(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	inDim := 4
	outDim := 4

	// Deterministic weights and bias
	W := [][]float64{
		{1, 2, 3, 4},
		{2, 0, 1, 1},
		{0, 1, 0, 1},
		{1, 1, 1, 1},
	}
	b := []float64{10, 20, 30, 40}
	input := []float64{1, 2, 3, 4}

	// --- Plaintext shadow model ---
	plainLinear := layers.NewLinear(inDim, outDim, false, nil)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			plainLinear.W.Data[j*inDim+i] = W[j][i]
		}
		plainLinear.B.Data[j] = b[j]
	}
	plainActivation, _ := layers.NewActivation("ReLU3", false, nil)
	inputTensor := tensor.New(inDim)
	copy(inputTensor.Data, input)
	plainOut, err := plainLinear.ForwardPlaintext(inputTensor)
	if err != nil {
		t.Fatalf("Plain Linear Forward error: %v", err)
	}
	plainActOut, err := plainActivation.Forward(plainOut)
	if err != nil {
		t.Fatalf("Plain Activation Forward error: %v", err)
	}
	plainActTensor := plainActOut.(*tensor.Tensor)

	// --- HE model ---
	heLinear := layers.NewLinear(inDim, outDim, true, he)
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			heLinear.W.Data[j*inDim+i] = W[j][i]
		}
		heLinear.B.Data[j] = b[j]
	}
	heLinear.SyncHE()
	heActivation, _ := layers.NewActivation("ReLU3", true, he)

	slots := he.Params.MaxSlots()
	xvec := make([]complex128, slots)
	for i := 0; i < inDim; i++ {
		xvec[i] = complex(input[i], 0)
	}
	ptX := ckks.NewPlaintext(he.Params, he.Params.MaxLevel())
	ptX.Scale = he.Params.DefaultScale()
	if err := he.Encoder.Encode(xvec, ptX); err != nil {
		t.Fatalf("encode error: %v", err)
	}
	ctX, err := he.Encryptor.EncryptNew(ptX)
	if err != nil {
		t.Fatalf("encrypt error: %v", err)
	}

	// Linear(HE) -> Activation(HE)
	ctY, err := heLinear.ForwardCipherMasked(ctX)
	if err != nil {
		t.Fatalf("Linear ForwardCipherMasked error: %v", err)
	}

	fmt.Printf("[Test] Before Refresh: Level=%d Scale=%v\n", ctY.Level(), ctY.Scale)
	ctY = he.Refresh(ctY)
	fmt.Printf("[Test] After Refresh: Level=%d Scale=%v\n", ctY.Level(), ctY.Scale)

	ctAct, err := heActivation.ForwardCipher(ctY)
	if err != nil {
		t.Fatalf("Activation ForwardCipher error: %v", err)
	}

	// Decrypt and compare
	ptAct := he.Decryptor.DecryptNew(ctAct)
	valsAct := make([]complex128, slots)
	he.Encoder.Decode(ptAct, valsAct)
	eps := 1e-3
	for j := 0; j < outDim; j++ {
		got := real(valsAct[j])
		want := plainActTensor.Data[j]
		if math.Abs(got-want) > eps {
			t.Errorf("slot %d: got %.6f want %.6f", j, got, want)
		} else {
			t.Logf("slot %d OK: got %.6f want %.6f", j, got, want)
		}
	}
}
