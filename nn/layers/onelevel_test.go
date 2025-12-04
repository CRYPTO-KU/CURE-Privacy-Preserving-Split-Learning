package layers

import (
	"cure_lib/core/ckkswrapper"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestOneLevelBatchMul_Toy4x4(t *testing.T) {
	heCtx := ckkswrapper.NewHeContext()
	// build a 4×4 matrix W and vector v of your choice
	W := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 16},
	}
	v := []float64{1, 2, 3, 4}
	// encrypt two batch vectors: columns 1&2 packed, columns 3&4 packed
	xBatch := encryptBatches(v, heCtx) // stub helper
	yCt, err := OneLevelBatchMul(W, xBatch, heCtx)
	require.NoError(t, err)
	y := decryptAndSumAll(yCt, heCtx) // sum slots after decryption
	fmt.Printf("[BatchMul] Decrypted slot vectors: %v\n", decryptSlotVectors(yCt, heCtx))
	fmt.Printf("[BatchMul] Aggregated output: %v\n", y)
	// check y equals W·plain_v
	expected := matVecMul(W, v)
	require.InEpsilonSlice(t, expected, y, 1e-3)
}

func TestOneLevelScalarMul_Toy4x4(t *testing.T) {
	heCtx := ckkswrapper.NewHeContext()
	W := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 16},
	}
	v := []float64{1, 2, 3, 4}
	xCt := encryptVector(v, heCtx) // stub helper
	yCt, err := OneLevelScalarMul(W, xCt, heCtx)
	require.NoError(t, err)
	y := decryptAndSumAll(yCt, heCtx)
	fmt.Printf("[ScalarMul] Decrypted slot vectors: %v\n", decryptSlotVectors(yCt, heCtx))
	fmt.Printf("[ScalarMul] Aggregated output: %v\n", y)
	expected := matVecMul(W, v)
	require.InEpsilonSlice(t, expected, y, 1e-3)
}

// --- Helper stubs ---

func encryptBatches(v []float64, heCtx *ckkswrapper.HeContext) []*rlwe.Ciphertext {
	slots := heCtx.Params.MaxSlots()
	// For toy 4x4, pack columns 1&2, 3&4 as two ciphertexts
	batches := [][]int{{0, 1}, {2, 3}}
	cts := make([]*rlwe.Ciphertext, len(batches))
	for i, batch := range batches {
		vec := make([]complex128, slots)
		for j, idx := range batch {
			vec[j] = complex(v[idx], 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		pt.Scale = heCtx.Params.DefaultScale()
		heCtx.Encoder.Encode(vec, pt)
		ct, err := heCtx.Encryptor.EncryptNew(pt)
		if err != nil {
			panic(err)
		}
		cts[i] = ct
	}
	return cts
}

func encryptVector(v []float64, heCtx *ckkswrapper.HeContext) *rlwe.Ciphertext {
	slots := heCtx.Params.MaxSlots()
	vec := make([]complex128, slots)
	for i := range v {
		vec[i] = complex(v[i], 0)
	}
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	pt.Scale = heCtx.Params.DefaultScale()
	heCtx.Encoder.Encode(vec, pt)
	ct, err := heCtx.Encryptor.EncryptNew(pt)
	if err != nil {
		panic(err)
	}
	return ct
}

func decryptAndSumAll(cts []*rlwe.Ciphertext, heCtx *ckkswrapper.HeContext) []float64 {
	res := make([]float64, len(cts))
	for i, ct := range cts {
		pt := heCtx.Decryptor.DecryptNew(ct)
		vec := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(pt, vec)
		sum := 0.0
		for j := 0; j < len(vec); j++ {
			sum += real(vec[j])
		}
		res[i] = sum
	}
	return res
}

func decryptSlotVectors(cts []*rlwe.Ciphertext, heCtx *ckkswrapper.HeContext) [][]float64 {
	res := make([][]float64, len(cts))
	for i, ct := range cts {
		pt := heCtx.Decryptor.DecryptNew(ct)
		vec := make([]complex128, heCtx.Params.MaxSlots())
		heCtx.Encoder.Decode(pt, vec)
		vals := make([]float64, 4) // Only first 4 slots for toy
		for j := 0; j < 4; j++ {
			vals[j] = real(vec[j])
		}
		res[i] = vals
	}
	return res
}

func matVecMul(W [][]float64, v []float64) []float64 {
	out := make([]float64, len(W))
	for i := range W {
		for j := range v {
			out[i] += W[i][j] * v[j]
		}
	}
	return out
}
