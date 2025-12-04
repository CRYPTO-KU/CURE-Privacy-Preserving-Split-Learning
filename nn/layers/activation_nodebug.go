//go:build !debug
// +build !debug

package layers

import "testing"

func (a *Activation) DebugCompareForward(result interface{}, t *testing.T) {}
