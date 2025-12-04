package layers

import "cure_lib/core/ckkswrapper"

// Conv1D simply wraps Conv2D with kh=1.
// Input  [C, L]        – no batch for simplicity
// Output [C_out, L-k+1]
func NewConv1D(inC, outC, k int, encrypted bool, heCtx *ckkswrapper.HeContext) *Conv2D {
	// Conv2D args: (inChan, outChan, kh, kw, encrypted, heCtx)
	return NewConv2D(inC, outC, 1, k, encrypted, heCtx)
}
