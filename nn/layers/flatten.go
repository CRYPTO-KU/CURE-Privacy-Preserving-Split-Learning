package layers

import (
	"cure_lib/tensor"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Flatten layer: reshapes tensor to 1D (plain), no-op for HE

type Flatten struct{ encrypted bool }

func NewFlatten(encrypted bool) *Flatten { return &Flatten{encrypted} }

// EnableEncrypted switches the layer between encrypted and plaintext mode
func (f *Flatten) EnableEncrypted(encrypted bool) {
	f.encrypted = encrypted
}

func (f *Flatten) Forward(x interface{}) (interface{}, error) {
	if f.encrypted {
		if ctSlice, ok := x.([]*rlwe.Ciphertext); ok {
			if len(ctSlice) == 1 {
				return ctSlice[0], nil
			}
			return nil, fmt.Errorf("Flatten: HE input slice length %d > 1 not supported yet", len(ctSlice))
		}
		return x, nil
	} // no-op for HE
	t := x.(*tensor.Tensor)
	y := tensor.New(len(t.Data))
	copy(y.Data, t.Data)
	return y, nil
}
func (f *Flatten) Backward(g interface{}) (interface{}, error) { return g, nil }
func (f *Flatten) Update(float64) error                        { return nil }
func (f *Flatten) Encrypted() bool                             { return f.encrypted }
func (f *Flatten) Levels() int                                 { return 0 }

// Interface methods for HE benchmarking
func (f *Flatten) ForwardHE(x interface{}) (interface{}, error) {
	if ctSlice, ok := x.([]*rlwe.Ciphertext); ok {
		if len(ctSlice) == 1 {
			return ctSlice[0], nil
		}
		return nil, fmt.Errorf("Flatten: HE input slice length %d > 1 not supported yet", len(ctSlice))
	}
	return x, nil
}

func (f *Flatten) BackwardHEIface(g interface{}) (interface{}, error) {
	return g, nil // no-op for HE
}

func (f *Flatten) Tag() string {
	return "Flatten"
}
