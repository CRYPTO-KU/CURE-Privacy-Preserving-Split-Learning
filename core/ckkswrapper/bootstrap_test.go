package ckkswrapper

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestCheatBootstrap(t *testing.T) {
	// Create HE context
	heCtx := NewHeContext()

	// Create test data
	data := make([]float64, heCtx.Params.MaxSlots())
	for i := range data {
		data[i] = float64(i) * 0.1
	}

	// Encrypt
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	if err := heCtx.Encoder.Encode(data, pt); err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	ct, err := heCtx.Encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	originalLevel := ct.Level()
	t.Logf("Original level: %d", originalLevel)

	// Perform cheat bootstrap
	refreshed, err := heCtx.CheatBootstrap(ct)
	if err != nil {
		t.Fatalf("CheatBootstrap failed: %v", err)
	}

	newLevel := refreshed.Level()
	t.Logf("Refreshed level: %d", newLevel)

	// Verify level is restored to max
	if newLevel != heCtx.Params.MaxLevel() {
		t.Errorf("Level = %d, want %d", newLevel, heCtx.Params.MaxLevel())
	}

	// Decrypt and verify data is preserved
	ptOut := heCtx.Decryptor.DecryptNew(refreshed)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(ptOut, decoded)

	// Check values match
	maxErr := 0.0
	for i := 0; i < 100; i++ {
		diff := math.Abs(real(decoded[i]) - data[i])
		if diff > maxErr {
			maxErr = diff
		}
	}

	t.Logf("Max error after bootstrap: %e", maxErr)
	if maxErr > 1e-6 {
		t.Errorf("Data corrupted after bootstrap, max error = %e", maxErr)
	}
}

func TestCheatBootstrapInPlace(t *testing.T) {
	heCtx := NewHeContext()

	data := make([]float64, heCtx.Params.MaxSlots())
	for i := range data {
		data[i] = float64(i%100) * 0.05
	}

	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	if err := heCtx.Encoder.Encode(data, pt); err != nil {
		t.Fatalf("Encode failed: %v", err)
	}
	ct, err := heCtx.Encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Bootstrap in place
	err = heCtx.CheatBootstrapInPlace(ct)
	if err != nil {
		t.Fatalf("CheatBootstrapInPlace failed: %v", err)
	}

	// Verify level is restored
	if ct.Level() != heCtx.Params.MaxLevel() {
		t.Errorf("Level = %d, want %d", ct.Level(), heCtx.Params.MaxLevel())
	}

	// Verify data
	ptOut := heCtx.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(ptOut, decoded)

	for i := 0; i < 50; i++ {
		expected := float64(i%100) * 0.05
		if math.Abs(real(decoded[i])-expected) > 1e-6 {
			t.Errorf("Data[%d] = %f, want %f", i, real(decoded[i]), expected)
		}
	}
}

func TestNeedsBootstrap(t *testing.T) {
	heCtx := NewHeContext()

	data := make([]float64, heCtx.Params.MaxSlots())
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(data, pt)
	ct, _ := heCtx.Encryptor.EncryptNew(pt)

	// Fresh ciphertext should not need bootstrap
	if NeedsBootstrap(ct, 1) {
		t.Errorf("Fresh ciphertext should not need bootstrap")
	}

	// Test with various thresholds
	maxLevel := heCtx.Params.MaxLevel()

	if !NeedsBootstrap(ct, maxLevel+1) {
		t.Errorf("Should need bootstrap when threshold > level")
	}

	// Test with level 0 threshold (default)
	if NeedsBootstrap(ct, 0) {
		// Should use threshold=1 by default
		if ct.Level() <= 1 {
			// This is correct behavior
		} else {
			t.Errorf("NeedsBootstrap with threshold=0 behaving unexpectedly")
		}
	}
}

func TestCheatBootstrapAfterOperations(t *testing.T) {
	heCtx := NewHeContextWithLogN(13)
	serverKit := heCtx.GenServerKit([]int{})

	// Create test data
	data := make([]float64, heCtx.Params.MaxSlots())
	for i := range data {
		data[i] = 0.5
	}

	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(data, pt)
	ct, _ := heCtx.Encryptor.EncryptNew(pt)

	initialLevel := ct.Level()
	t.Logf("Initial level: %d", initialLevel)

	// Perform some multiplications to consume levels
	for i := 0; i < 3; i++ {
		serverKit.Evaluator.MulRelin(ct, ct, ct)
		serverKit.Evaluator.Rescale(ct, ct)
		t.Logf("Level after mul %d: %d", i+1, ct.Level())
	}

	levelAfterOps := ct.Level()
	t.Logf("Level after operations: %d", levelAfterOps)

	// Bootstrap
	refreshed, err := heCtx.CheatBootstrap(ct)
	if err != nil {
		t.Fatalf("CheatBootstrap failed: %v", err)
	}

	t.Logf("Level after bootstrap: %d", refreshed.Level())

	// Should be back to max level
	if refreshed.Level() != heCtx.Params.MaxLevel() {
		t.Errorf("Level = %d, want %d", refreshed.Level(), heCtx.Params.MaxLevel())
	}
}
