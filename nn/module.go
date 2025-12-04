package nn

import (
	"cure_lib/tensor"
)

// Module defines a single layer/unit in the network.
type Module interface {
	Forward(input interface{}) (interface{}, error)
	// Backward computes gradients and propagates them.
	// It takes the gradient of the loss with respect to the module's output,
	// and returns the gradient of the loss with respect to the module's input.
	Backward(gradOut interface{}) (interface{}, error)
	Encrypted() bool
	Levels() int
}

// Sequential chains multiple Modules in order.
type Sequential struct {
	Layers []Module
}

// Forward applies each layer in sequence (plaintext-only version).
func (s *Sequential) Forward(x *tensor.Tensor) (interface{}, error) {
	var err error
	var out interface{} = x
	for _, layer := range s.Layers {
		out, err = layer.Forward(out)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Backward applies Backward in reverse order (plaintext-only version).
func (s *Sequential) Backward(grad *tensor.Tensor) (interface{}, error) {
	var err error
	var out interface{} = grad
	for i := len(s.Layers) - 1; i >= 0; i-- {
		out, err = s.Layers[i].Backward(out)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Levels sums Levels() of all layers.
func (s *Sequential) Levels() int {
	sum := 0
	for _, layer := range s.Layers {
		sum += layer.Levels()
	}
	return sum
}

// Encrypted returns true if any layer is encrypted.
func (s *Sequential) Encrypted() bool {
	for _, layer := range s.Layers {
		if layer.Encrypted() {
			return true
		}
	}
	return false
}
