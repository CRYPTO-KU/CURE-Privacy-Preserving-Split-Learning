package layers

import (
	"cure_lib/tensor"
	"testing"
)

func TestFlatten_Plain(t *testing.T) {
	f := NewFlatten(false)
	input := tensor.New(2, 3)
	for i := range input.Data {
		input.Data[i] = float64(i)
	}
	out, err := f.Forward(input)
	if err != nil {
		t.Fatalf("flatten error: %v", err)
	}
	flat := out.(*tensor.Tensor)
	if len(flat.Data) != 6 {
		t.Fatalf("flatten wrong size")
	}
}
