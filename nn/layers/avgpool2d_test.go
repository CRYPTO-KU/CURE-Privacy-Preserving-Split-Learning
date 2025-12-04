package layers

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestAvgPool2D_PlainVsReference(t *testing.T) {
	C, H, W, p := 2, 4, 4, 2
	x := tensor.New(C, H, W)
	for i := range x.Data {
		x.Data[i] = rand.Float64()
	}
	layer := NewAvgPool2D(p, false, nil)
	out, err := layer.ForwardPlain(x)
	if err != nil {
		t.Fatalf("ForwardPlain failed: %v", err)
	}
	// Reference: compute manually
	ref := tensor.New(C, H/p, W/p)
	for c := 0; c < C; c++ {
		for oh := 0; oh < H/p; oh++ {
			for ow := 0; ow < W/p; ow++ {
				sum := 0.0
				for ph := 0; ph < p; ph++ {
					for pw := 0; pw < p; pw++ {
						sum += x.Data[(c*H+(oh*p+ph))*W+(ow*p+pw)]
					}
				}
				ref.Data[(c*(H/p)+oh)*(W/p)+ow] = sum / float64(p*p)
			}
		}
	}
	for i := range out.Data {
		if abs(out.Data[i]-ref.Data[i]) > 1e-8 {
			t.Errorf("Mismatch at %d: got %f, want %f", i, out.Data[i], ref.Data[i])
		}
	}
}

func TestAvgPool2D_HEForwardCorrectness(t *testing.T) {
	C, H, W, p := 2, 4, 4, 2
	heCtx := ckkswrapper.NewHeContext()
	layer := NewAvgPool2D(p, true, heCtx)
	_ = layer.SetDimensions(H, W)
	_ = layer.SyncHE()
	// Deterministic input
	x := tensor.New(C, H, W)
	for i := range x.Data {
		x.Data[i] = float64(i + 1)
	}
	fmt.Printf("[TEST] Input tensor (C=%d, H=%d, W=%d):\n", C, H, W)
	for c := 0; c < C; c++ {
		fmt.Printf("  Channel %d:\n", c)
		for h := 0; h < H; h++ {
			fmt.Printf("    ")
			for w := 0; w < W; w++ {
				fmt.Printf("%5.1f ", x.Data[c*H*W+h*W+w])
			}
			fmt.Printf("\n")
		}
	}
	// Encrypt each channel
	cts := make([]*rlwe.Ciphertext, C)
	for c := 0; c < C; c++ {
		vec := make([]complex128, heCtx.Params.MaxSlots())
		for i := 0; i < H*W; i++ {
			vec[i] = complex(x.Data[c*H*W+i], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		pt.Scale = heCtx.Params.DefaultScale()
		heCtx.Encoder.Encode(vec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		cts[c] = ct
		// Print packed input slots
		dec := heCtx.Decryptor.DecryptNew(ct)
		raw := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(dec, raw)
		fmt.Printf("[TEST] Channel %d packed slots: %v\n", c, raw[:H*W])
	}
	// Forward HE
	y_cts, err := layer.ForwardHE(cts)
	if err != nil {
		t.Fatalf("ForwardHE failed: %v", err)
	}
	// Decrypt
	y_he := tensor.New(C, H/p, W/p)
	for c := 0; c < C; c++ {
		pt := heCtx.Decryptor.DecryptNew(y_cts[c])
		decoded := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(pt, decoded)
		fmt.Printf("[TEST] Channel %d HE output slots: %v\n", c, decoded[:(H/p)*(W/p)])
		for i := 0; i < (H/p)*(W/p); i++ {
			y_he.Data[c*(H/p)*(W/p)+i] = real(decoded[i])
		}
	}
	// Plain reference
	layerPlain := NewAvgPool2D(p, false, nil)
	ref, _ := layerPlain.ForwardPlain(x)
	fmt.Printf("[TEST] Reference output:\n")
	for c := 0; c < C; c++ {
		fmt.Printf("  Channel %d: ", c)
		for i := 0; i < (H/p)*(W/p); i++ {
			fmt.Printf("%7.3f ", ref.Data[c*(H/p)*(W/p)+i])
		}
		fmt.Printf("\n")
	}
	for i := range y_he.Data {
		fmt.Printf("[TEST] Output slot %d: HE=%7.3f, REF=%7.3f\n", i, y_he.Data[i], ref.Data[i])
		if abs(y_he.Data[i]-ref.Data[i]) > 1e-3 {
			t.Errorf("HE mismatch at %d: got %f, want %f", i, y_he.Data[i], ref.Data[i])
		}
	}
}

func TestAvgPool2D_AllOnes_HE(t *testing.T) {
	// build a 4×4 input where every entry = 1
	C, H, W, p := 1, 4, 4, 2
	heCtx := ckkswrapper.NewHeContext()
	ones := make([]float64, C*H*W)
	for i := range ones {
		ones[i] = 1
	}
	in := []*rlwe.Ciphertext{packAndEncrypt(ones, H, W, heCtx)}
	layer := NewAvgPool2D(p, true, heCtx)
	_ = layer.SetDimensions(H, W)
	_ = layer.SyncHE()
	yHE, err := layer.ForwardHE(in)
	require.NoError(t, err)
	out := decryptAndUnpack(yHE[0], C, H/p, W/p, heCtx)
	// every slot should be ≈1
	for i, v := range out {
		if abs(v-1.0) > 1e-3 {
			t.Errorf("slot %d: got %v, want 1.0", i, v)
		}
	}
}

func TestAvgPool2D_Placement(t *testing.T) {
	// ----- deterministic 4×4 input, 2×2 pool -----
	C, H, W, p := 2, 4, 4, 2
	in := tensor.New(C, H, W)
	for c := 0; c < C; c++ {
		for i := 0; i < H*W; i++ {
			in.Data[c*H*W+i] = float64(c*H*W + i + 1) // 1..16, 17..32
		}
	}

	// Plain reference
	plainRef, _ := NewAvgPool2D(p, false, nil).ForwardPlain(in)

	// HE setup
	heCtx := ckkswrapper.NewHeContext()
	layer := NewAvgPool2D(p, true, heCtx)
	_ = layer.SetDimensions(H, W)
	layer.SyncHE()

	// encrypt channel-wise
	cts := make([]*rlwe.Ciphertext, C)
	for c := 0; c < C; c++ {
		vec := make([]complex128, heCtx.Params.MaxSlots())
		copy(vec, complexify(in.Data[c*H*W:(c+1)*H*W]))
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		pt.Scale = heCtx.Params.DefaultScale()
		heCtx.Encoder.Encode(vec, pt)
		cts[c], _ = heCtx.Encryptor.EncryptNew(pt)
	}

	// HE forward
	outs, err := layer.ForwardHE(cts)
	if err != nil {
		t.Fatalf("HE forward: %v", err)
	}

	// decrypt & compare slot-by-slot
	outH, outW := H/p, W/p
	wantSlots := outH * outW
	for c := 0; c < C; c++ {
		dec := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(heCtx.Decryptor.DecryptNew(outs[c]), dec)
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				k := oh*outW + ow
				got := real(dec[k])
				want := plainRef.Data[c*outH*outW+k]
				if !approxEqual(got, want, 0.01) {
					t.Fatalf("c=%d (oh,ow)=(%d,%d) slot=%d got %.4f want %.4f",
						c, oh, ow, k, got, want)
				}
			}
		}
		// sanity: all slots beyond wanted region should be ~0
		for k := wantSlots; k < 8; k++ {
			if math.Abs(real(dec[k])) > 1e-3 {
				t.Fatalf("unexpected residue in slot %d: %.4f", k, real(dec[k]))
			}
		}
	}
}

func TestAvgPool2D_HEBackwardScatter(t *testing.T) {
	// Validate that BackwardHE scatters each output grad equally to its p×p window
	H, W, p := 4, 4, 2
	outH, outW := H/p, W/p

	heCtx := ckkswrapper.NewHeContext()
	layer := NewAvgPool2D(p, true, heCtx)
	_ = layer.SetDimensions(H, W)
	_ = layer.SyncHE()

	// Gradient at pooled outputs: [1,2,3,4] in slots 0..3
	grads := make([]float64, outH*outW)
	for k := 0; k < outH*outW; k++ {
		grads[k] = float64(k + 1)
	}
	vec := make([]complex128, heCtx.Params.MaxSlots())
	for i := 0; i < outH*outW; i++ {
		vec[i] = complex(grads[i], 0)
	}
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	pt.Scale = heCtx.Params.DefaultScale()
	heCtx.Encoder.Encode(vec, pt)
	ct, _ := heCtx.Encryptor.EncryptNew(pt)

	// Run backward
	dIn, err := layer.BackwardHE([]*rlwe.Ciphertext{ct})
	require.NoError(t, err)

	// Decrypt and compare against expected scatter result
	dec := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(heCtx.Decryptor.DecryptNew(dIn[0]), dec)

	expected := make([]float64, H*W)
	k := 0
	for by := 0; by < H; by += p {
		for bx := 0; bx < W; bx += p {
			g := grads[k] / float64(p*p)
			for dy := 0; dy < p; dy++ {
				for dx := 0; dx < p; dx++ {
					idx := (by+dy)*W + (bx + dx)
					expected[idx] = g
				}
			}
			k++
		}
	}

	for i := 0; i < H*W; i++ {
		if math.Abs(real(dec[i])-expected[i]) > 1e-2 {
			t.Fatalf("slot %d: got %.4f, want %.4f", i, real(dec[i]), expected[i])
		}
	}
}

// packAndEncrypt packs a real vector into a ciphertext.
func packAndEncrypt(vec []float64, H, W int, heCtx *ckkswrapper.HeContext) *rlwe.Ciphertext {
	slots := heCtx.Params.MaxSlots()
	cvec := make([]complex128, slots)
	for i := 0; i < H*W && i < len(vec); i++ {
		cvec[i] = complex(vec[i], 0)
	}
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	pt.Scale = heCtx.Params.DefaultScale()
	heCtx.Encoder.Encode(cvec, pt)
	ct, _ := heCtx.Encryptor.EncryptNew(pt)
	return ct
}

// decryptSlot0 decrypts a ciphertext and returns the real part of slot 0.
func decryptSlot0(ct *rlwe.Ciphertext, heCtx *ckkswrapper.HeContext) float64 {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)
	return real(decoded[0])
}

// decryptAndUnpack decrypts a ciphertext and returns the real parts of the first C*H*W slots.
func decryptAndUnpack(ct *rlwe.Ciphertext, C, H, W int, heCtx *ckkswrapper.HeContext) []float64 {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)
	out := make([]float64, C*H*W)
	for i := 0; i < C*H*W; i++ {
		out[i] = real(decoded[i])
	}
	return out
}

// decryptSlotk decrypts a ciphertext and returns the real part of slot k.
func decryptSlotk(ct *rlwe.Ciphertext, k int, heCtx *ckkswrapper.HeContext) float64 {
	pt := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(pt, decoded)
	return real(decoded[k])
}

// requireNoError is a helper for error checking.
func requireNoError(t *testing.T, err error) {
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func complexify(v []float64) []complex128 {
	out := make([]complex128, len(v))
	for i, x := range v {
		out[i] = complex(x, 0)
	}
	return out
}
func approxEqual(a, b, eps float64) bool { return math.Abs(a-b) < eps }
