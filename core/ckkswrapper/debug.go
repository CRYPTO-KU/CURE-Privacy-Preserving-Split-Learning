//go:build debug
// +build debug

package ckkswrapper

import (
	"cure_lib/tensor"
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// DebugCompare compares a ciphertext result with a shadow plaintext tensor
// and reports any divergence beyond the specified tolerance
func (he *HeContext) DebugCompare(ct *rlwe.Ciphertext, shadow *tensor.Tensor, label string, tolerance float64, t *testing.T) {
	if t == nil {
		return // Skip if no testing context provided
	}

	// Decrypt the ciphertext
	pt := he.Decryptor.DecryptNew(ct)

	// Decode into float64 slice
	decoded := make([]complex128, he.Params.MaxSlots())
	he.Encoder.Decode(pt, decoded)

	// Compare with shadow tensor
	maxDiff := 0.0
	maxDiffIdx := -1

	for i := 0; i < len(shadow.Data) && i < len(decoded); i++ {
		heVal := real(decoded[i])
		shadowVal := shadow.Data[i]
		diff := math.Abs(heVal - shadowVal)

		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}

		if diff > tolerance {
			t.Errorf("%s: Divergence at index %d: HE=%f, Shadow=%f, Diff=%f",
				label, i, heVal, shadowVal, diff)
		}
	}

	// Log the comparison result
	if maxDiff <= tolerance {
		t.Logf("%s: ✓ Max difference: %f at index %d", label, maxDiff, maxDiffIdx)
	} else {
		t.Logf("%s: ✗ Max difference: %f at index %d (exceeds tolerance %f)",
			label, maxDiff, maxDiffIdx, tolerance)
	}

	// Check for numerical instability
	for i := 0; i < len(shadow.Data) && i < len(decoded); i++ {
		heVal := real(decoded[i])
		shadowVal := shadow.Data[i]

		if math.Abs(heVal) > 100.0 || math.Abs(shadowVal) > 100.0 {
			t.Logf("WARNING: Potential numerical instability detected at %s index %d: HE=%f, Shadow=%f",
				label, i, heVal, shadowVal)
		}
	}
}
