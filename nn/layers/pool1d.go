package layers

import (
	"cure_lib/tensor"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type MaxPool1D struct {
	Window int
}

// --- Plain ---
func (p *MaxPool1D) Forward(x interface{}) (interface{}, error) {
	t, ok := x.(*tensor.Tensor)
	if !ok {
		return nil, ErrType
	}

	// Handle 4D [B, C, 1, W] (from Conv1D wrapping Conv2D)
	if len(t.Shape) == 4 && t.Shape[2] == 1 {
		B, C, W := t.Shape[0], t.Shape[1], t.Shape[3]
		outW := W / p.Window
		out := tensor.New(B, C, 1, outW)

		for b := 0; b < B; b++ {
			for c := 0; c < C; c++ {
				for i := 0; i < outW; i++ {
					// Base index for this channel/batch
					baseIdx := b*C*W + c*W
					// Output index
					outIdx := b*C*outW + c*outW + i

					max := t.Data[baseIdx+i*p.Window]
					for j := 1; j < p.Window; j++ {
						idx := baseIdx + i*p.Window + j
						if idx < len(t.Data) && t.Data[idx] > max {
							max = t.Data[idx]
						}
					}
					out.Data[outIdx] = max
				}
			}
		}
		return out, nil
	}

	// Fallback to 2D [C, L]
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("MaxPool1D expects 2D [C, L] or 4D [B, C, 1, W], got shape %v", t.Shape)
	}

	C, L := t.Shape[0], t.Shape[1]
	outL := L / p.Window
	out := tensor.New(C, outL)
	for c := 0; c < C; c++ {
		for i := 0; i < outL; i++ {
			max := t.Data[c*L+i*p.Window]
			for j := 1; j < p.Window; j++ {
				idx := c*L + i*p.Window + j
				if idx < len(t.Data) && t.Data[idx] > max {
					max = t.Data[idx]
				}
			}
			out.Data[c*outL+i] = max
		}
	}
	return out, nil
}
func (p *MaxPool1D) Backward(g interface{}) (interface{}, error) { return g, nil }
func (p *MaxPool1D) Update(lr float64) error                     { return nil }
func (p *MaxPool1D) Encrypted() bool                             { return false }
func (p *MaxPool1D) Levels() int                                 { return 0 }
func (p *MaxPool1D) Tag() string {
	return fmt.Sprintf("MaxPool1D_%d", p.Window)
}

// --- HE stubs (compile-time only) ---
func (p *MaxPool1D) ForwardHE(in []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	// just return a shallow copy; timing ~0
	out := make([]*rlwe.Ciphertext, len(in))
	copy(out, in)
	return out, nil
}
func (p *MaxPool1D) BackwardHE(in []*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error) {
	out := make([]*rlwe.Ciphertext, len(in))
	copy(out, in)
	return out, nil
}

func NewMaxPool1D(window int) *MaxPool1D { return &MaxPool1D{Window: window} }

var ErrType = &TypeError{"input must be *tensor.Tensor"}

type TypeError struct{ msg string }

func (e *TypeError) Error() string { return e.msg }
