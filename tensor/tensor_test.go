package tensor

import "testing"

func TestNewShape(t *testing.T) {
	t1 := New(2, 3)
	if len(t1.Data) != 6 {
		t.Fatalf("expected 6 elements, got %d", len(t1.Data))
	}
	if len(t1.Shape) != 2 || t1.Shape[0] != 2 || t1.Shape[1] != 3 {
		t.Fatalf("unexpected shape: %v", t1.Shape)
	}
}

func TestAdd(t *testing.T) {
	a := &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}}
	b := &Tensor{Data: []float64{4, 5, 6}, Shape: []int{3}}
	c, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{5, 7, 9}
	for i := range want {
		if c.Data[i] != want[i] {
			t.Errorf("at %d, got %f, want %f", i, c.Data[i], want[i])
		}
	}
}

func TestMatMul(t *testing.T) {
	a := &Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}}
	b := &Tensor{Data: []float64{5, 6, 7, 8}, Shape: []int{2, 2}}
	c, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{19, 22, 43, 50}
	for i := range want {
		if c.Data[i] != want[i] {
			t.Errorf("at %d, got %f, want %f", i, c.Data[i], want[i])
		}
	}
}

func TestReluPlain(t *testing.T) {
	a := &Tensor{Data: []float64{-1, 0, 3}, Shape: []int{3}}
	c := ReluPlain(a)
	want := []float64{0, 0, 3}
	for i := range want {
		if c.Data[i] != want[i] {
			t.Errorf("at %d, got %f, want %f", i, c.Data[i], want[i])
		}
	}
}
