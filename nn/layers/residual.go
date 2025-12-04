package layers

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/tensor"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type residualSubModule interface {
	Forward(input interface{}) (interface{}, error)
	Backward(gradOut interface{}) (interface{}, error)
	Encrypted() bool
	Levels() int
	Update(lr float64) error
	Tag() string
}

type ResidualBlock struct {
	Main  []residualSubModule
	heCtx *ckkswrapper.HeContext
}

func NewResidualBlock(mods []residualSubModule, heCtx *ckkswrapper.HeContext) *ResidualBlock {
	return &ResidualBlock{Main: mods, heCtx: heCtx}
}

func (r *ResidualBlock) Forward(x interface{}) (interface{}, error) {
	in := x
	var err error
	for _, m := range r.Main {
		in, err = m.Forward(in)
		if err != nil {
			return nil, err
		}
	}
	// add skip: assume ciphertext or tensor – rely on AddNew / tensor.Add
	switch v := in.(type) {
	case *rlwe.Ciphertext:
		if conv, ok := r.Main[0].(*Conv2D); ok {
			return conv.serverKit.Evaluator.AddNew(v, x.(*rlwe.Ciphertext))
		}
		return nil, fmt.Errorf("ResidualBlock: first module is not Conv2D for HE skip connection")
	case *tensor.Tensor:
		return tensor.Add(v, x.(*tensor.Tensor))
	default:
		return nil, fmt.Errorf("unsupported type %T", v)
	}
}

func (r *ResidualBlock) Backward(g interface{}) (interface{}, error) {
	grad := g
	var err error
	for i := len(r.Main) - 1; i >= 0; i-- {
		grad, err = r.Main[i].Backward(grad)
		if err != nil {
			return nil, err
		}
	}
	// return grad accumulated from both paths (= 2×grad for identity)
	switch v := grad.(type) {
	case *rlwe.Ciphertext:
		if conv, ok := r.Main[0].(*Conv2D); ok {
			return conv.serverKit.Evaluator.AddNew(v, g.(*rlwe.Ciphertext))
		}
		return nil, fmt.Errorf("ResidualBlock: first module is not Conv2D for HE skip connection")
	case *tensor.Tensor:
		return tensor.Add(v, g.(*tensor.Tensor))
	default:
		return nil, fmt.Errorf("unsupported type %T", v)
	}
}

func (r *ResidualBlock) Update(lr float64) error {
	for _, m := range r.Main {
		if err := m.Update(lr); err != nil {
			return err
		}
	}
	return nil
}

func (r *ResidualBlock) Encrypted() bool {
	return r.Main[0].Encrypted()
}

func (r *ResidualBlock) Levels() int {
	levels := 0
	for _, m := range r.Main {
		levels += m.Levels()
	}
	return levels
}

func (r *ResidualBlock) Tag() string {
	tags := "ResidualBlock["
	for i, m := range r.Main {
		if i > 0 {
			tags += ","
		}
		tags += m.Tag()
	}
	tags += "]"
	return tags
}
