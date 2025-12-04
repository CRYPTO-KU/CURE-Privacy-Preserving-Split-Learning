package layers

import (
	"testing"

	"cure_lib/core/ckkswrapper"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func BenchmarkOneLevelBatchMul(b *testing.B) {
	heCtx := ckkswrapper.NewHeContext()
	W, xBatch := randomMatrixAndBatches(512, 512, heCtx)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		OneLevelBatchMul(W, xBatch, heCtx)
	}
}

func BenchmarkOneLevelScalarMul(b *testing.B) {
	heCtx := ckkswrapper.NewHeContext()
	W, xVec := randomMatrixAndVector(512, heCtx)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		OneLevelScalarMul(W, xVec, heCtx)
	}
}

// --- Helper stubs ---
func randomMatrixAndBatches(rows, cols int, heCtx *ckkswrapper.HeContext) ([][]float64, []*rlwe.Ciphertext) {
	// TODO: Implement random matrix and batch encryption for benchmark
	return nil, nil
}

func randomMatrixAndVector(rows int, heCtx *ckkswrapper.HeContext) ([][]float64, *rlwe.Ciphertext) {
	// TODO: Implement random matrix and vector encryption for benchmark
	return nil, nil
}
