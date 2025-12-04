package bench

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/nn"
	"cure_lib/nn/layers"
)

// BuiltNet holds a model's layers and which are HE-capable
type BuiltNet struct {
	Name   string
	Layers []nn.Module // Only the list of layers is needed
}

// 1. 784-128-32-10 (2-layer FC)
func BuildMNISTFC(heCtx *ckkswrapper.HeContext, encrypted bool) BuiltNet {
	layersList := []nn.Module{
		layers.NewLinear(784, 128, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(128, 32, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(32, 10, encrypted, heCtx),
	}
	return BuiltNet{
		Name:   "mnistfc",
		Layers: layersList,
	}
}

// 2. Classic LeNet (conv-pool-conv-pool-2FC)
func BuildLeNet(heCtx *ckkswrapper.HeContext, encrypted bool) BuiltNet {
	layersList := []nn.Module{
		layers.NewConv2D(1, 6, 5, 5, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		// AvgPool2D with p=2 (2x2 pooling)
		NewAvgPool2DModule(2, encrypted, heCtx),
		layers.NewConv2D(6, 16, 5, 5, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		NewAvgPool2DModule(2, encrypted, heCtx),
		layers.NewFlatten(encrypted),
		layers.NewLinear(400, 120, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(120, 84, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(84, 10, encrypted, heCtx),
	}
	return BuiltNet{
		Name:   "lenet",
		Layers: layersList,
	}
}

// AvgPool2DModule wraps AvgPool2D to implement nn.Module
type AvgPool2DModule struct {
	pool      *layers.AvgPool2D
	encrypted bool
}

func NewAvgPool2DModule(p int, encrypted bool, heCtx *ckkswrapper.HeContext) *AvgPool2DModule {
	return &AvgPool2DModule{
		pool:      layers.NewAvgPool2D(p, encrypted, heCtx),
		encrypted: encrypted,
	}
}

func (m *AvgPool2DModule) Forward(x interface{}) (interface{}, error) {
	return m.pool.ForwardHEIface(x)
}

func (m *AvgPool2DModule) Backward(grad interface{}) (interface{}, error) {
	return m.pool.BackwardHEIface(grad)
}

func (m *AvgPool2DModule) Encrypted() bool {
	return m.encrypted
}

func (m *AvgPool2DModule) Levels() int {
	return m.pool.Levels()
}

// 3. 64-32-16-10 BCW
func BuildBCWFC(heCtx *ckkswrapper.HeContext, encrypted bool) BuiltNet {
	layersList := []nn.Module{
		layers.NewLinear(64, 32, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(32, 16, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(16, 10, encrypted, heCtx),
	}
	return BuiltNet{
		Name:   "bcwfc",
		Layers: layersList,
	}
}

// 4. 1-D-CNN
func BuildAudio1D(heCtx *ckkswrapper.HeContext, encrypted bool) BuiltNet {
	layersList := []nn.Module{
		layers.NewConv1D(12, 16, 3, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx), // LeakyReLU approx
		layers.NewMaxPool1D(2),                       // always plaintext
		layers.NewConv1D(16, 8, 3, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewMaxPool1D(2),
		layers.NewFlatten(encrypted),
		layers.NewLinear(2000, 5, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx), // Use ReLU3 instead of Softmax for timing
	}
	return BuiltNet{
		Name:   "audio1d",
		Layers: layersList,
	}
}

// 5. ResNet block (first conv & 1 residual block)
func BuildResNetBlock(heCtx *ckkswrapper.HeContext, encrypted bool) BuiltNet {
	layersList := []nn.Module{
		layers.NewConv2D(3, 64, 7, 7, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewConv2D(64, 64, 3, 3, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewConv2D(64, 64, 3, 3, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
	}
	return BuiltNet{
		Name:   "resnet",
		Layers: layersList,
	}
}

// Helper to panic on activation creation error
func mustNewActivation(name string, encrypted bool, heCtx *ckkswrapper.HeContext) nn.Module {
	act, err := layers.NewActivation(name, encrypted, heCtx)
	if err != nil {
		panic(err)
	}
	return act
}
