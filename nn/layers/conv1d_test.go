package layers

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"
	"testing"
)

func TestConv1D_Plain(t *testing.T) {
	he := ckkswrapper.NewHeContext()
	conv := NewConv1D(1, 1, 3, false, he)
	in := tensor.New(1, 1, 8)
	for i := 0; i < 8; i++ {
		in.Data[i] = float64(i + 1)
	}
	out, err := conv.ForwardPlain(in)
	if err != nil {
		t.Fatalf("bad conv1d: %v", err)
	}
	if len(out.Shape) != 4 || out.Shape[0] != 1 || out.Shape[1] != 1 || out.Shape[2] != 1 || out.Shape[3] != 6 {
		t.Fatalf("unexpected output shape: %v", out.Shape)
	}
}
