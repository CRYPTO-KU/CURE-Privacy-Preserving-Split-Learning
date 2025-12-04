package ckkswrapper

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// CheatBootstrap refreshes a ciphertext's level by decrypting and re-encrypting.
// This is a "cheating" bootstrap that requires the secret key - used for
// development and testing. In production, use real bootstrapping.
//
// The refreshed ciphertext will have the maximum level and default scale.
func (h *HeContext) CheatBootstrap(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	// Decrypt to plaintext
	pt := h.Decryptor.DecryptNew(ct)

	// Decode the values
	values := make([]complex128, h.Params.MaxSlots())
	if err := h.Encoder.Decode(pt, values); err != nil {
		return nil, err
	}

	// Re-encode at max level
	newPt := ckks.NewPlaintext(h.Params, h.Params.MaxLevel())
	if err := h.Encoder.Encode(values, newPt); err != nil {
		return nil, err
	}

	// Re-encrypt
	return h.Encryptor.EncryptNew(newPt)
}

// CheatBootstrapInPlace refreshes a ciphertext in place.
func (h *HeContext) CheatBootstrapInPlace(ct *rlwe.Ciphertext) error {
	refreshed, err := h.CheatBootstrap(ct)
	if err != nil {
		return err
	}
	*ct = *refreshed
	return nil
}

// NeedsBootstrap returns true if the ciphertext level is at or below the threshold.
// Default threshold is 1 level remaining.
func NeedsBootstrap(ct *rlwe.Ciphertext, threshold int) bool {
	if threshold <= 0 {
		threshold = 1
	}
	return ct.Level() <= threshold
}
