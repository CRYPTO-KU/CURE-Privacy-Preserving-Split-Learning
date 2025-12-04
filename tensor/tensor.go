package tensor

import "fmt"

// Tensor is a simple n-D array backed by a flat []float64.
type Tensor struct {
	Data  []float64
	Shape []int
}

// New allocates a Tensor of given shape (product of dims = len(Data)).
func New(shape ...int) *Tensor {
	// Compute total size
	total := 1
	for _, d := range shape {
		total *= d
	}
	return &Tensor{
		Data:  make([]float64, total),
		Shape: append([]int(nil), shape...),
	}
}

// NewWithData creates a 1-D tensor from existing data slice.
func NewWithData(data []float64) *Tensor {
	return &Tensor{
		Data:  append([]float64(nil), data...),
		Shape: []int{len(data)},
	}
}

// Add returns a+b (same shape), or error if shapes differ.
func Add(a, b *Tensor) (*Tensor, error) {
	// Shapes must match
	if len(a.Shape) != len(b.Shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape, b.Shape)
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return nil, fmt.Errorf("shape mismatch: %v vs %v", a.Shape, b.Shape)
		}
	}
	// Element-wise add
	out := New(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out, nil
}

// MatMul returns a×b (2-D only), or error if dims mismatch.
func MatMul(a, b *Tensor) (*Tensor, error) {
	// Only 2-D tensors
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("MatMul requires 2-D tensors, got %v and %v", a.Shape, b.Shape)
	}
	r, k := a.Shape[0], a.Shape[1]
	k2, c := b.Shape[0], b.Shape[1]
	if k != k2 {
		return nil, fmt.Errorf("inner dimensions must match: %d vs %d", k, k2)
	}
	out := New(r, c)
	// Compute C = A×B
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum := 0.0
			for t := 0; t < k; t++ {
				sum += a.Data[i*k+t] * b.Data[t*c+j]
			}
			out.Data[i*c+j] = sum
		}
	}
	return out, nil
}

// ReluPlain applies ReLU to each element in a, returns new Tensor.
func ReluPlain(a *Tensor) *Tensor {
	out := New(a.Shape...)
	for i, v := range a.Data {
		if v > 0 {
			out.Data[i] = v
		} else {
			out.Data[i] = 0
		}
	}
	return out
}

// At returns the element at the given indices.
// For a 4D tensor [a, b, c, d], At(i, j, k, l) returns the element at position [i][j][k][l].
func (t *Tensor) At(indices ...int) float64 {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("At: expected %d indices, got %d", len(t.Shape), len(indices)))
	}

	// Compute linear index
	idx := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			panic(fmt.Sprintf("At: index %d out of bounds for dimension %d (shape: %v)", indices[i], i, t.Shape))
		}
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}

	return t.Data[idx]
}

// Set sets the element at the given indices to the given value.
func (t *Tensor) Set(value float64, indices ...int) {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("Set: expected %d indices, got %d", len(t.Shape), len(indices)))
	}

	// Compute linear index
	idx := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			panic(fmt.Sprintf("Set: index %d out of bounds for dimension %d (shape: %v)", indices[i], i, t.Shape))
		}
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}

	t.Data[idx] = value
}
