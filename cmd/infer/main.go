// cure-infer: Encrypted inference using saved weights
package main

import (
	"encoding/json"
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

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

var (
	weightsFile = flag.String("weights", "", "Weights JSON file")
	inputFile   = flag.String("input", "", "Input JSON file")
	logN        = flag.Int("logN", 13, "Ring dimension log2")
	encrypted   = flag.Bool("encrypted", true, "Use HE encryption")
	verbose     = flag.Bool("verbose", true, "Verbose output")
	topK        = flag.Int("topk", 3, "Top predictions to show")
)

func main() {
	flag.Parse()
	utils.Verbose = *verbose

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║              CURE_lib Encrypted Inference                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	if *weightsFile == "" {
		fmt.Println("\nNo weights file. Running demo mode...")
		runDemo()
		return
	}

	// Load weights
	weights, err := utils.LoadWeights(*weightsFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading weights: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d layers\n", len(weights.Layers))

	// Load input
	var inputData []float64
	if *inputFile != "" {
		data, _ := os.ReadFile(*inputFile)
		json.Unmarshal(data, &inputData)
	} else {
		inputData = make([]float64, 784)
		for i := range inputData {
			inputData[i] = rand.Float64()
		}
	}
	fmt.Printf("Input dim: %d\n", len(inputData))

	// Initialize HE
	var heCtx *ckkswrapper.HeContext
	if *encrypted {
		heCtx = ckkswrapper.NewHeContextWithLogN(*logN)
	}

	// Build model from weights
	model := buildFromWeights(weights, heCtx, *encrypted)

	// Run inference
	fmt.Println("\nRunning inference...")
	start := time.Now()
	predictions, err := runInference(model, inputData, heCtx, *encrypted)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Time: %.4fs\n", time.Since(start).Seconds())

	// Show results
	showResults(predictions, *topK)
}

func runDemo() {
	fmt.Printf("LogN: %d, Encrypted: %v\n", *logN, *encrypted)

	var heCtx *ckkswrapper.HeContext
	if *encrypted {
		heCtx = ckkswrapper.NewHeContextWithLogN(*logN)
	}

	// Demo model
	model := []nn.Module{
		layers.NewLinear(784, 128, *encrypted, heCtx),
		mustAct("ReLU3", *encrypted, heCtx),
		layers.NewLinear(128, 32, *encrypted, heCtx),
		mustAct("ReLU3", *encrypted, heCtx),
		layers.NewLinear(32, 10, *encrypted, heCtx),
	}

	for _, l := range model {
		if lin, ok := l.(*layers.Linear); ok {
			for i := range lin.W.Data {
				lin.W.Data[i] = rand.Float64() * 0.01
			}
			if *encrypted {
				lin.SyncHE()
			}
		}
	}

	// Random input
	inputData := make([]float64, 784)
	for i := range inputData {
		inputData[i] = rand.Float64()
	}

	start := time.Now()
	predictions, _ := runInference(model, inputData, heCtx, *encrypted)
	fmt.Printf("Inference time: %.4fs\n", time.Since(start).Seconds())

	showResults(predictions, *topK)
}

func runInference(model []nn.Module, inputData []float64, heCtx *ckkswrapper.HeContext, encrypted bool) ([]float64, error) {
	if encrypted {
		return runHE(model, inputData, heCtx)
	}
	return runPlain(model, inputData)
}

func runHE(model []nn.Module, inputData []float64, heCtx *ckkswrapper.HeContext) ([]float64, error) {
	// Encrypt
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	inputC := make([]complex128, heCtx.Params.MaxSlots())
	for i := 0; i < len(inputData) && i < len(inputC); i++ {
		inputC[i] = complex(inputData[i], 0)
	}
	heCtx.Encoder.Encode(inputC, pt)
	ct, _ := heCtx.Encryptor.EncryptNew(pt)

	// Forward
	var out interface{} = ct
	var err error
	for _, layer := range model {
		if ctOut, ok := out.(*rlwe.Ciphertext); ok && ckkswrapper.NeedsBootstrap(ctOut, 2) {
			out, _ = heCtx.CheatBootstrap(ctOut)
		}
		out, err = layer.Forward(out)
		if err != nil {
			return nil, err
		}
	}

	// Decrypt
	ctOut := out.(*rlwe.Ciphertext)
	ptOut := heCtx.Decryptor.DecryptNew(ctOut)
	decoded := make([]complex128, heCtx.Params.MaxSlots())
	heCtx.Encoder.Decode(ptOut, decoded)

	result := make([]float64, 10)
	for i := 0; i < 10; i++ {
		result[i] = real(decoded[i])
	}
	return result, nil
}

func runPlain(model []nn.Module, inputData []float64) ([]float64, error) {
	input := tensor.NewWithData(inputData)
	var out interface{} = input
	var err error
	for _, layer := range model {
		out, err = layer.Forward(out)
		if err != nil {
			return nil, err
		}
	}
	return out.(*tensor.Tensor).Data, nil
}

func buildFromWeights(weights *utils.ModelWeights, heCtx *ckkswrapper.HeContext, encrypted bool) []nn.Module {
	var model []nn.Module
	for _, lw := range weights.Layers {
		if lw.Weight != nil {
			lin := layers.NewLinear(lw.Weight.Shape[1], lw.Weight.Shape[0], encrypted, heCtx)
			copy(lin.W.Data, lw.Weight.Data)
			if lw.Bias != nil {
				copy(lin.B.Data, lw.Bias.Data)
			}
			if encrypted {
				lin.SyncHE()
			}
			model = append(model, lin)
			act, _ := layers.NewActivation("ReLU3", encrypted, heCtx)
			model = append(model, act)
		}
	}
	return model
}

func mustAct(name string, encrypted bool, heCtx *ckkswrapper.HeContext) nn.Module {
	act, _ := layers.NewActivation(name, encrypted, heCtx)
	return act
}

func showResults(predictions []float64, k int) {
	probs := nn.Softmax(tensor.NewWithData(predictions))
	indices := topKIndices(predictions, k)

	fmt.Printf("\nTop %d predictions:\n", k)
	for i, idx := range indices {
		fmt.Printf("  %d. Class %d: %.4f\n", i+1, idx, probs.Data[idx])
	}
}

func topKIndices(vals []float64, k int) []int {
	if k > len(vals) {
		k = len(vals)
	}
	indices := make([]int, k)
	used := make(map[int]bool)
	for i := 0; i < k; i++ {
		maxIdx, maxVal := -1, math.Inf(-1)
		for j, v := range vals {
			if !used[j] && v > maxVal {
				maxVal, maxIdx = v, j
			}
		}
		indices[i] = maxIdx
		used[maxIdx] = true
	}
	return indices
}
