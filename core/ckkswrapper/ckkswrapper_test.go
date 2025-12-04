package ckkswrapper

import (
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestHeContextRoundTrip(t *testing.T) {
	h := NewHeContext()
	vals := []complex128{3.1415926535}
	slots := h.Params.MaxSlots()
	pt := ckks.NewPlaintext(h.Params, h.Params.MaxLevel())
	err := h.Encoder.Encode(vals, pt)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}
	ct, err := h.Encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("encrypt error: %v", err)
	}
	gotPt := h.Decryptor.DecryptNew(ct)
	decoded := make([]complex128, slots)
	err = h.Encoder.Decode(gotPt, decoded)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if diff := real(decoded[0]) - real(vals[0]); diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("roundtrip mismatch: got %f, want %f", real(decoded[0]), real(vals[0]))
	}

	kit := h.GenServerKit([]int{1, 2, -1})
	ct2, err := kit.Evaluator.MulNew(ct, ct)
	if err != nil {
		t.Fatalf("evaluator MulNew error: %v", err)
	}
	_ = ct2
}
