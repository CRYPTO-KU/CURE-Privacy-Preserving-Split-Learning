// cure-train: Standalone single-process trainer for CURE_lib
//
// Usage:
//
//	cure-train --model=mnist --epochs=10 --lr=0.01 --logN=13
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn"
	"cure_lib/nn/layers"
	"cure_lib/tensor"
	"cure_lib/utils"
)

var (
	modelType    = flag.String("model", "mnist", "Model type: mnist, bcw")
	epochs       = flag.Int("epochs", 5, "Number of training epochs")
	learningRate = flag.Float64("lr", 0.01, "Learning rate")
	logN         = flag.Int("logN", 13, "Ring dimension log2 (13-16)")
	encrypted    = flag.Bool("encrypted", true, "Use HE encryption")
	verbose      = flag.Bool("verbose", true, "Verbose output")
	seed         = flag.Int64("seed", 42, "Random seed")
	samples      = flag.Int("samples", 100, "Number of synthetic samples")
	outputFile   = flag.String("output", "", "Output weights file (JSON)")
)

func main() {
	flag.Parse()
	utils.Verbose = *verbose
	rand.Seed(*seed)

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    CURE_lib Trainer                          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("\nConfiguration:\n")
	fmt.Printf("  Model:         %s\n", *modelType)
	fmt.Printf("  Epochs:        %d\n", *epochs)
	fmt.Printf("  Learning Rate: %.4f\n", *learningRate)
	fmt.Printf("  LogN:          %d\n", *logN)
	fmt.Printf("  Encrypted:     %v\n", *encrypted)
	fmt.Printf("  Samples:       %d\n", *samples)
	fmt.Println()

	// Initialize HE context if encrypted
	var heCtx *ckkswrapper.HeContext
	if *encrypted {
		fmt.Println("Initializing HE context...")
		start := time.Now()
		heCtx = ckkswrapper.NewHeContextWithLogN(*logN)
		fmt.Printf("HE initialization: %.2fs\n", time.Since(start).Seconds())
	}

	// Build model
	model := buildModel(*modelType, heCtx, *encrypted)
	fmt.Printf("\nModel: %d layers\n", len(model.Layers))

	// Generate synthetic data
	fmt.Printf("Generating %d synthetic samples...\n", *samples)
	inputs, labels := generateData(model.InputDim, model.OutputDim, *samples)

	// Training loop
	fmt.Println("\nStarting training...")
	stats := &utils.TimingStats{}
	totalStart := time.Now()

	for epoch := 0; epoch < *epochs; epoch++ {
		epochStart := time.Now()
		epochLoss := 0.0

		for i := 0; i < len(inputs); i++ {
			loss, err := trainStep(model, inputs[i], labels[i], *learningRate, stats)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error at sample %d: %v\n", i, err)
				continue
			}
			epochLoss += loss
		}

		avgLoss := epochLoss / float64(len(inputs))
		fmt.Printf("Epoch %d/%d | Loss: %.6f | Time: %.2fs\n",
			epoch+1, *epochs, avgLoss, time.Since(epochStart).Seconds())
	}

	stats.TotalTime = time.Since(totalStart)
	fmt.Printf("\nTraining complete! Total time: %.2fs\n", stats.TotalTime.Seconds())

	if *verbose {
		utils.PrintTimingStats(stats, *epochs*len(inputs))
	}

	// Save weights
	if *outputFile != "" {
		fmt.Printf("\nSaving weights to %s...\n", *outputFile)
		if err := saveWeights(model, *outputFile); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Done!")
	}
}

type Model struct {
	Layers    []nn.Module
	InputDim  int
	OutputDim int
}

func buildModel(modelType string, heCtx *ckkswrapper.HeContext, encrypted bool) *Model {
	var layerList []nn.Module
	var inputDim, outputDim int

	switch modelType {
	case "mnist":
		inputDim, outputDim = 784, 10
		layerList = []nn.Module{
			layers.NewLinear(784, 128, encrypted, heCtx),
			mustAct("ReLU3", encrypted, heCtx),
			layers.NewLinear(128, 32, encrypted, heCtx),
			mustAct("ReLU3", encrypted, heCtx),
			layers.NewLinear(32, 10, encrypted, heCtx),
		}
	case "bcw":
		inputDim, outputDim = 64, 10
		layerList = []nn.Module{
			layers.NewLinear(64, 32, encrypted, heCtx),
			mustAct("ReLU3", encrypted, heCtx),
			layers.NewLinear(32, 16, encrypted, heCtx),
			mustAct("ReLU3", encrypted, heCtx),
			layers.NewLinear(16, 10, encrypted, heCtx),
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown model: %s\n", modelType)
		os.Exit(1)
	}

	// Initialize weights
	for _, layer := range layerList {
		if lin, ok := layer.(*layers.Linear); ok {
			scale := math.Sqrt(2.0 / float64(lin.W.Shape[1]+lin.W.Shape[0]))
			for i := range lin.W.Data {
				lin.W.Data[i] = rand.NormFloat64() * scale
			}
			if encrypted {
				lin.SyncHE()
			}
		}
	}

	return &Model{Layers: layerList, InputDim: inputDim, OutputDim: outputDim}
}

func mustAct(name string, encrypted bool, heCtx *ckkswrapper.HeContext) nn.Module {
	act, err := layers.NewActivation(name, encrypted, heCtx)
	if err != nil {
		panic(err)
	}
	return act
}

func generateData(inputDim, outputDim, n int) ([]*tensor.Tensor, []*tensor.Tensor) {
	inputs := make([]*tensor.Tensor, n)
	labels := make([]*tensor.Tensor, n)
	for i := 0; i < n; i++ {
		inputs[i] = tensor.New(inputDim)
		for j := range inputs[i].Data {
			inputs[i].Data[j] = rand.NormFloat64()
		}
		labels[i] = tensor.New(outputDim)
		labels[i].Data[rand.Intn(outputDim)] = 1.0
	}
	return inputs, labels
}

func trainStep(model *Model, input, label *tensor.Tensor, lr float64, stats *utils.TimingStats) (float64, error) {
	// Forward
	start := time.Now()
	var out interface{} = input
	var err error
	for _, layer := range model.Layers {
		out, err = layer.Forward(out)
		if err != nil {
			return 0, err
		}
	}
	stats.ForwardPassTime += time.Since(start)

	// Loss
	outTensor := out.(*tensor.Tensor)
	probs := nn.Softmax(outTensor)
	loss := 0.0
	for i := 0; i < model.OutputDim; i++ {
		if label.Data[i] > 0 {
			p := probs.Data[i]
			if p < 1e-10 {
				p = 1e-10
			}
			loss -= label.Data[i] * math.Log(p)
		}
	}

	// Backward
	start = time.Now()
	grad := tensor.New(model.OutputDim)
	for i := 0; i < model.OutputDim; i++ {
		grad.Data[i] = probs.Data[i] - label.Data[i]
	}
	var gradOut interface{} = grad
	for i := len(model.Layers) - 1; i >= 0; i-- {
		gradOut, err = model.Layers[i].Backward(gradOut)
		if err != nil {
			return 0, err
		}
	}
	stats.BackwardPassTime += time.Since(start)

	return loss, nil
}

func saveWeights(model *Model, filepath string) error {
	weights := &utils.ModelWeights{
		Version: "1.0",
		Layers:  make(map[string]utils.LayerWeight),
	}
	for i, layer := range model.Layers {
		if lin, ok := layer.(*layers.Linear); ok {
			weights.Layers[fmt.Sprintf("linear_%d", i)] = utils.LayerWeight{
				Weight: utils.TensorToWeightData("weight", lin.W),
				Bias:   utils.TensorToWeightData("bias", lin.B),
			}
		}
	}
	return utils.SaveWeights(filepath, weights)
}
