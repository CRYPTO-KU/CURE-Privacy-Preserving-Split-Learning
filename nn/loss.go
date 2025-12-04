package nn

import (
	"cure_lib/tensor"
	"math"
)

type CrossEntropyLoss struct{}

// Backward computes the gradient of the cross-entropy loss with softmax.
// grad = (softmax_output - one_hot_label)
func (c *CrossEntropyLoss) Backward(softmaxOut, oneHotLabel *tensor.Tensor) *tensor.Tensor {
	grad := tensor.New(len(softmaxOut.Data))
	for i := range grad.Data {
		grad.Data[i] = softmaxOut.Data[i] - oneHotLabel.Data[i]
	}
	return grad
}

// Softmax applies the softmax function to a tensor.
func Softmax(logits *tensor.Tensor) *tensor.Tensor {
	maxLogit := logits.Data[0]
	for _, v := range logits.Data {
		if v > maxLogit {
			maxLogit = v
		}
	}
	expSum := 0.0
	exps := make([]float64, len(logits.Data))
	for i, v := range logits.Data {
		e := math.Exp(v - maxLogit)
		exps[i] = e
		expSum += e
	}
	softmax := tensor.New(len(logits.Data))
	for i, e := range exps {
		softmax.Data[i] = e / expSum
	}
	return softmax
}
