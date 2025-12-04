package layers

import (
	"cure_lib/core/ckkswrapper"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func encryptRandomInput() []*rlwe.Ciphertext {
	H, W := 4, 4
	C := 2
	heCtx := ckkswrapper.NewHeContext()
	slots := heCtx.Params.MaxSlots()
	cts := make([]*rlwe.Ciphertext, C)
	for c := 0; c < C; c++ {
		vec := make([]complex128, slots)
		for i := 0; i < H*W; i++ {
			vec[i] = complex(rand.Float64(), 0)
		}
		pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
		pt.Scale = heCtx.Params.DefaultScale()
		heCtx.Encoder.Encode(vec, pt)
		ct, _ := heCtx.Encryptor.EncryptNew(pt)
		cts[c] = ct
	}
	return cts
}

func BenchmarkAvgPool2D_HE(b *testing.B) {
	H, W, p := 4, 4, 2
	heCtx := ckkswrapper.NewHeContext()
	layer := NewAvgPool2D(p, true, heCtx)
	layer.SetDimensions(H, W)
	layer.SyncHE()
	cts := encryptRandomInput()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = layer.ForwardHE(cts)
	}
}
