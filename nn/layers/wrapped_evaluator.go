package layers

import (
	"fmt"

	"cure_lib/utils"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// WrappedEvaluator wraps a ckks.Evaluator to count operations
type WrappedEvaluator struct {
	eval *ckks.Evaluator

	// Operation counters
	RotateCount  int
	MulCount     int
	RelinCount   int
	RescaleCount int
	AddCount     int
}

// NewWrappedEvaluator creates a new wrapped evaluator
func NewWrappedEvaluator(eval *ckks.Evaluator) *WrappedEvaluator {
	return &WrappedEvaluator{
		eval: eval,
	}
}

// ResetCounters resets all operation counters to zero
func (w *WrappedEvaluator) ResetCounters() {
	w.RotateCount = 0
	w.MulCount = 0
	w.RelinCount = 0
	w.RescaleCount = 0
	w.AddCount = 0
}

// PrintCounters prints the current operation counts.
// Respects utils.Verbose flag - does nothing if Verbose is false.
func (w *WrappedEvaluator) PrintCounters(phaseName string) {
	if !utils.Verbose {
		return
	}
	fmt.Fprintf(utils.Output, "=== Phase: %s ===\n", phaseName)
	fmt.Fprintf(utils.Output, "Rotates: %d, Muls: %d, Relins: %d, Rescales: %d, Adds: %d\n",
		w.RotateCount, w.MulCount, w.RelinCount, w.RescaleCount, w.AddCount)
}

// RotateNew wraps eval.RotateNew and counts rotations
func (w *WrappedEvaluator) RotateNew(ct *rlwe.Ciphertext, krot int) (*rlwe.Ciphertext, error) {
	w.RotateCount++
	return w.eval.RotateNew(ct, krot)
}

// MulNew wraps eval.MulNew and counts multiplications
func (w *WrappedEvaluator) MulNew(ct *rlwe.Ciphertext, pt *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
	w.MulCount++
	return w.eval.MulNew(ct, pt)
}

// RelinearizeNew wraps eval.RelinearizeNew and counts relinearizations
func (w *WrappedEvaluator) RelinearizeNew(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	w.RelinCount++
	return w.eval.RelinearizeNew(ct)
}

// Rescale wraps eval.Rescale and counts rescales
func (w *WrappedEvaluator) Rescale(ct *rlwe.Ciphertext, ctOut *rlwe.Ciphertext) error {
	w.RescaleCount++
	return w.eval.Rescale(ct, ctOut)
}

// AddNew wraps eval.AddNew and counts additions
func (w *WrappedEvaluator) AddNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	w.AddCount++
	return w.eval.AddNew(ct1, ct2)
}

// SubNew wraps eval.SubNew and counts as addition
func (w *WrappedEvaluator) SubNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	w.AddCount++ // Subtraction is similar cost to addition
	return w.eval.SubNew(ct1, ct2)
}
