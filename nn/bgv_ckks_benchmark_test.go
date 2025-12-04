package nn

import (
	"fmt"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// BenchmarkResult stores timing results for a single operation
type BenchmarkResult struct {
	Operation string
	BGVTime   time.Duration
	CKKSTime  time.Duration
	Speedup   float64 // BGVTime / CKKSTime (>1 means CKKS is faster)
}

// TestBGVvsCKKSPrimitives compares BGV and CKKS performance on primitive operations
func TestBGVvsCKKSPrimitives(t *testing.T) {
	const numIterations = 100

	t.Log("\n════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	t.Log("BGV vs CKKS PRIMITIVE OPERATIONS BENCHMARK")
	t.Logf("Lattigo v6 | Comparable Parameters | %d iterations per operation", numIterations)
	t.Log("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")

	// ========================================================================
	// SETUP BGV
	// ========================================================================
	t.Log("\nInitializing BGV scheme...")
	bgvParams, err := bgv.NewParametersFromLiteral(bgv.ParametersLiteral{
		LogN:             14,                        // Ring dimension N = 2^14 = 16384
		LogQ:             []int{55, 45, 45, 45, 45}, // Modulus chain
		LogP:             []int{55},                 // Special prime for key switching
		PlaintextModulus: 65537,                     // Prime plaintext modulus
	})
	if err != nil {
		t.Fatalf("BGV params failed: %v", err)
	}

	bgvKgen := rlwe.NewKeyGenerator(bgvParams)
	bgvSk, bgvPk := bgvKgen.GenKeyPairNew()
	bgvRlk := bgvKgen.GenRelinearizationKeyNew(bgvSk)

	// Generate rotation keys for BGV
	bgvGaloisKeys := bgvKgen.GenGaloisKeysNew([]uint64{
		bgvParams.GaloisElement(1),
		bgvParams.GaloisElement(2),
		bgvParams.GaloisElement(4),
		bgvParams.GaloisElement(8),
	}, bgvSk)

	bgvEvk := rlwe.NewMemEvaluationKeySet(bgvRlk, bgvGaloisKeys...)
	bgvEncoder := bgv.NewEncoder(bgvParams)
	bgvEncryptor := rlwe.NewEncryptor(bgvParams, bgvPk)
	bgvEvaluator := bgv.NewEvaluator(bgvParams, bgvEvk)

	t.Logf("BGV: LogN=%d, LogQ=%v, LogP=%v, PlaintextModulus=%d",
		bgvParams.LogN(), bgvParams.LogQ(), bgvParams.LogP(), bgvParams.PlaintextModulus())

	// ========================================================================
	// SETUP CKKS
	// ========================================================================
	t.Log("\nInitializing CKKS scheme...")
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            14,                        // Same ring dimension
		LogQ:            []int{55, 45, 45, 45, 45}, // Same modulus chain
		LogP:            []int{55},                 // Same special prime
		LogDefaultScale: 45,                        // Default scale for encoding
	})
	if err != nil {
		t.Fatalf("CKKS params failed: %v", err)
	}

	ckksKgen := rlwe.NewKeyGenerator(ckksParams)
	ckksSk, ckksPk := ckksKgen.GenKeyPairNew()
	ckksRlk := ckksKgen.GenRelinearizationKeyNew(ckksSk)

	// Generate rotation keys for CKKS
	ckksGaloisKeys := ckksKgen.GenGaloisKeysNew([]uint64{
		ckksParams.GaloisElement(1),
		ckksParams.GaloisElement(2),
		ckksParams.GaloisElement(4),
		ckksParams.GaloisElement(8),
	}, ckksSk)

	ckksEvk := rlwe.NewMemEvaluationKeySet(ckksRlk, ckksGaloisKeys...)
	ckksEncoder := ckks.NewEncoder(ckksParams)
	ckksEncryptor := rlwe.NewEncryptor(ckksParams, ckksPk)
	ckksEvaluator := ckks.NewEvaluator(ckksParams, ckksEvk)

	t.Logf("CKKS: LogN=%d, LogQ=%v, LogP=%v, LogDefaultScale=%d",
		ckksParams.LogN(), ckksParams.LogQ(), ckksParams.LogP(), 45)

	// ========================================================================
	// CREATE TEST DATA
	// ========================================================================
	slots := bgvParams.N() // BGV uses N slots for integers

	// BGV test vectors (integers mod plaintext modulus)
	bgvVec1 := make([]uint64, slots)
	bgvVec2 := make([]uint64, slots)
	for i := range bgvVec1 {
		bgvVec1[i] = uint64(i % 100)
		bgvVec2[i] = uint64((i + 50) % 100)
	}

	// CKKS test vectors (complex numbers, using real parts only)
	ckksSlots := ckksParams.MaxSlots()
	ckksVec1 := make([]complex128, ckksSlots)
	ckksVec2 := make([]complex128, ckksSlots)
	for i := range ckksVec1 {
		ckksVec1[i] = complex(float64(i%100), 0)
		ckksVec2[i] = complex(float64((i+50)%100), 0)
	}

	// Encode and encrypt BGV vectors
	bgvPt1 := bgv.NewPlaintext(bgvParams, bgvParams.MaxLevel())
	bgvPt2 := bgv.NewPlaintext(bgvParams, bgvParams.MaxLevel())
	bgvEncoder.Encode(bgvVec1, bgvPt1)
	bgvEncoder.Encode(bgvVec2, bgvPt2)
	bgvCt1, _ := bgvEncryptor.EncryptNew(bgvPt1)
	bgvCt2, _ := bgvEncryptor.EncryptNew(bgvPt2)

	// Encode and encrypt CKKS vectors
	ckksPt1 := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())
	ckksPt2 := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())
	ckksEncoder.Encode(ckksVec1, ckksPt1)
	ckksEncoder.Encode(ckksVec2, ckksPt2)
	ckksCt1, _ := ckksEncryptor.EncryptNew(ckksPt1)
	ckksCt2, _ := ckksEncryptor.EncryptNew(ckksPt2)

	results := make([]BenchmarkResult, 0)

	// ========================================================================
	// BENCHMARK: ADDITION (Ciphertext + Ciphertext)
	// ========================================================================
	t.Log("\nBenchmarking Addition (CT + CT)...")

	// BGV Addition
	start := time.Now()
	for i := 0; i < numIterations; i++ {
		bgvEvaluator.Add(bgvCt1, bgvCt2, bgvCt1)
	}
	bgvAddTime := time.Since(start) / time.Duration(numIterations)

	// CKKS Addition
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		ckksEvaluator.Add(ckksCt1, ckksCt2, ckksCt1)
	}
	ckksAddTime := time.Since(start) / time.Duration(numIterations)

	results = append(results, BenchmarkResult{
		Operation: "Addition (CT+CT)",
		BGVTime:   bgvAddTime,
		CKKSTime:  ckksAddTime,
		Speedup:   float64(bgvAddTime) / float64(ckksAddTime),
	})

	// Re-encrypt for next test
	bgvCt1, _ = bgvEncryptor.EncryptNew(bgvPt1)
	bgvCt2, _ = bgvEncryptor.EncryptNew(bgvPt2)
	ckksCt1, _ = ckksEncryptor.EncryptNew(ckksPt1)
	ckksCt2, _ = ckksEncryptor.EncryptNew(ckksPt2)

	// ========================================================================
	// BENCHMARK: MULTIPLICATION (Ciphertext * Ciphertext)
	// ========================================================================
	t.Log("Benchmarking Multiplication (CT * CT)...")

	// BGV Multiplication (without relinearization)
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		bgvEvaluator.MulNew(bgvCt1, bgvCt2)
	}
	bgvMulTime := time.Since(start) / time.Duration(numIterations)

	// CKKS Multiplication (without relinearization)
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		ckksEvaluator.MulNew(ckksCt1, ckksCt2)
	}
	ckksMulTime := time.Since(start) / time.Duration(numIterations)

	results = append(results, BenchmarkResult{
		Operation: "Multiplication (CT*CT)",
		BGVTime:   bgvMulTime,
		CKKSTime:  ckksMulTime,
		Speedup:   float64(bgvMulTime) / float64(ckksMulTime),
	})

	// ========================================================================
	// BENCHMARK: MULTIPLICATION + RELINEARIZATION
	// ========================================================================
	t.Log("Benchmarking Mul + Relinearization...")

	// BGV Mul + Relin
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		tmp, _ := bgvEvaluator.MulNew(bgvCt1, bgvCt2)
		bgvEvaluator.Relinearize(tmp, tmp)
	}
	bgvMulRelinTime := time.Since(start) / time.Duration(numIterations)

	// CKKS Mul + Relin
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		tmp, _ := ckksEvaluator.MulNew(ckksCt1, ckksCt2)
		ckksEvaluator.Relinearize(tmp, tmp)
	}
	ckksMulRelinTime := time.Since(start) / time.Duration(numIterations)

	results = append(results, BenchmarkResult{
		Operation: "Mul + Relinearization",
		BGVTime:   bgvMulRelinTime,
		CKKSTime:  ckksMulRelinTime,
		Speedup:   float64(bgvMulRelinTime) / float64(ckksMulRelinTime),
	})

	// ========================================================================
	// BENCHMARK: FULL MULTIPLICATION PIPELINE (Mul + Relin + Rescale for CKKS)
	// ========================================================================
	t.Log("Benchmarking Full Mul Pipeline (Mul+Relin vs Mul+Relin+Rescale)...")

	// Re-encrypt
	bgvCt1, _ = bgvEncryptor.EncryptNew(bgvPt1)
	bgvCt2, _ = bgvEncryptor.EncryptNew(bgvPt2)
	ckksCt1, _ = ckksEncryptor.EncryptNew(ckksPt1)
	ckksCt2, _ = ckksEncryptor.EncryptNew(ckksPt2)

	// BGV Full Mul (just Mul + Relin, no rescale needed)
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		bgvEvaluator.MulRelinNew(bgvCt1, bgvCt2)
	}
	bgvFullMulTime := time.Since(start) / time.Duration(numIterations)

	// CKKS Full Mul (Mul + Relin + Rescale)
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		tmp, _ := ckksEvaluator.MulRelinNew(ckksCt1, ckksCt2)
		ckksEvaluator.Rescale(tmp, tmp)
	}
	ckksFullMulTime := time.Since(start) / time.Duration(numIterations)

	results = append(results, BenchmarkResult{
		Operation: "Full Mul Pipeline",
		BGVTime:   bgvFullMulTime,
		CKKSTime:  ckksFullMulTime,
		Speedup:   float64(bgvFullMulTime) / float64(ckksFullMulTime),
	})

	// ========================================================================
	// BENCHMARK: ROTATION
	// ========================================================================
	t.Log("Benchmarking Rotation (by 1)...")

	// Re-encrypt fresh ciphertexts
	bgvCt1, _ = bgvEncryptor.EncryptNew(bgvPt1)
	ckksCt1, _ = ckksEncryptor.EncryptNew(ckksPt1)

	// BGV Rotation
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		bgvEvaluator.RotateColumnsNew(bgvCt1, 1)
	}
	bgvRotTime := time.Since(start) / time.Duration(numIterations)

	// CKKS Rotation
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		ckksEvaluator.RotateNew(ckksCt1, 1)
	}
	ckksRotTime := time.Since(start) / time.Duration(numIterations)

	results = append(results, BenchmarkResult{
		Operation: "Rotation (by 1)",
		BGVTime:   bgvRotTime,
		CKKSTime:  ckksRotTime,
		Speedup:   float64(bgvRotTime) / float64(ckksRotTime),
	})

	// ========================================================================
	// BENCHMARK: PLAINTEXT MULTIPLICATION
	// ========================================================================
	t.Log("Benchmarking Plaintext Multiplication (CT * PT)...")

	// Re-encrypt
	bgvCt1, _ = bgvEncryptor.EncryptNew(bgvPt1)
	ckksCt1, _ = ckksEncryptor.EncryptNew(ckksPt1)

	// BGV CT * PT
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		bgvEvaluator.MulNew(bgvCt1, bgvPt2)
	}
	bgvPtMulTime := time.Since(start) / time.Duration(numIterations)

	// CKKS CT * PT
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		ckksEvaluator.MulNew(ckksCt1, ckksPt2)
	}
	ckksPtMulTime := time.Since(start) / time.Duration(numIterations)

	results = append(results, BenchmarkResult{
		Operation: "Plaintext Mul (CT*PT)",
		BGVTime:   bgvPtMulTime,
		CKKSTime:  ckksPtMulTime,
		Speedup:   float64(bgvPtMulTime) / float64(ckksPtMulTime),
	})

	// ========================================================================
	// BENCHMARK: KEY SWITCHING (via rotation as proxy)
	// ========================================================================
	t.Log("Benchmarking Key Switching (rotation by 8)...")

	// Re-encrypt
	bgvCt1, _ = bgvEncryptor.EncryptNew(bgvPt1)
	ckksCt1, _ = ckksEncryptor.EncryptNew(ckksPt1)

	// BGV Key Switch (larger rotation)
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		bgvEvaluator.RotateColumnsNew(bgvCt1, 8)
	}
	bgvKsTime := time.Since(start) / time.Duration(numIterations)

	// CKKS Key Switch (larger rotation)
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		ckksEvaluator.RotateNew(ckksCt1, 8)
	}
	ckksKsTime := time.Since(start) / time.Duration(numIterations)

	results = append(results, BenchmarkResult{
		Operation: "Key Switch (rot by 8)",
		BGVTime:   bgvKsTime,
		CKKSTime:  ckksKsTime,
		Speedup:   float64(bgvKsTime) / float64(ckksKsTime),
	})

	// ========================================================================
	// PRINT RESULTS TABLE
	// ========================================================================
	t.Log("\n════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	t.Log("BENCHMARK RESULTS")
	t.Log("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	t.Logf("\n%-25s | %-15s | %-15s | %-15s | %-10s",
		"Operation", "BGV Time", "CKKS Time", "Difference", "Faster")
	t.Log("----------------------------------------------------------------------------------------------------------------------------")

	for _, r := range results {
		var faster string
		var diff string
		if r.Speedup > 1.0 {
			faster = "CKKS"
			diff = fmt.Sprintf("%.2fx faster", r.Speedup)
		} else if r.Speedup < 1.0 {
			faster = "BGV"
			diff = fmt.Sprintf("%.2fx faster", 1.0/r.Speedup)
		} else {
			faster = "SAME"
			diff = "1.00x"
		}
		t.Logf("%-25s | %-15s | %-15s | %-15s | %-10s",
			r.Operation,
			r.BGVTime.String(),
			r.CKKSTime.String(),
			diff,
			faster)
	}

	t.Log("----------------------------------------------------------------------------------------------------------------------------")

	// Calculate averages
	var avgSpeedup float64
	for _, r := range results {
		avgSpeedup += r.Speedup
	}
	avgSpeedup /= float64(len(results))

	t.Log("\n════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	t.Log("SUMMARY")
	t.Log("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	t.Logf("Parameters: LogN=14, LogQ=[55,45,45,45,45], LogP=[55]")
	t.Logf("Iterations per operation: %d", numIterations)
	t.Logf("Average speedup ratio (BGV/CKKS): %.2fx", avgSpeedup)
	if avgSpeedup > 1.0 {
		t.Logf("Overall: CKKS is on average %.2fx FASTER than BGV", avgSpeedup)
	} else {
		t.Logf("Overall: BGV is on average %.2fx FASTER than CKKS", 1.0/avgSpeedup)
	}
	t.Log("")
	t.Log("NOTE: Both schemes use the same ring dimension and modulus chain.")
	t.Log("      BGV operates on integers mod plaintext modulus (exact arithmetic).")
	t.Log("      CKKS operates on approximate complex/real numbers.")
	t.Log("      Performance differences are primarily due to encoding/NTT vs FFT operations.")
	t.Log("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
}
