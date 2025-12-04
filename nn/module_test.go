package nn

import (
	"cure_lib/tensor"
	"errors"
	"testing"
)

// dummy layer: adds a constant
type addLayer struct{ c float64 }

func (l *addLayer) Forward(input interface{}) (interface{}, error) {
	x, ok := input.(*tensor.Tensor)
	if !ok {
		return nil, errors.New("addLayer expects *tensor.Tensor input")
	}
	out, _ := tensor.Add(x, &tensor.Tensor{Data: []float64{l.c}, Shape: []int{1}})
	return out, nil
}
func (l *addLayer) Backward(gradOut interface{}) (interface{}, error) {
	grad, ok := gradOut.(*tensor.Tensor)
	if !ok {
		return nil, errors.New("addLayer expects *tensor.Tensor gradOut")
	}
	return grad, nil
}
func (l *addLayer) Levels() int     { return 1 }
func (l *addLayer) Encrypted() bool { return false }

// dummy layer: error on forward
type errLayer struct{}

func (l *errLayer) Forward(input interface{}) (interface{}, error) {
	return nil, errors.New("fail")
}
func (l *errLayer) Backward(gradOut interface{}) (interface{}, error) {
	return nil, nil
}
func (l *errLayer) Levels() int     { return 0 }
func (l *errLayer) Encrypted() bool { return true }

func TestSequentialPlain(t *testing.T) {
	a := tensor.New(1)
	a.Data[0] = 1
	seq := &Sequential{Layers: []Module{&addLayer{c: 2}, &addLayer{c: 3}}}
	outAny, err := seq.Forward(a)
	if err != nil {
		t.Fatal(err)
	}
	out, ok := outAny.(*tensor.Tensor)
	if !ok {
		t.Fatalf("expected *tensor.Tensor output, got %T", outAny)
	}
	if out.Data[0] != 6 {
		t.Fatalf("expected 6, got %f", out.Data[0])
	}
}

func TestSequentialLevelsEncrypted(t *testing.T) {
	seq := &Sequential{Layers: []Module{&addLayer{c: 0}, &errLayer{}}}
	if seq.Levels() != 1 {
		t.Errorf("expected Levels=1, got %d", seq.Levels())
	}
	if !seq.Encrypted() {
		t.Errorf("expected Encrypted=true")
	}
}
