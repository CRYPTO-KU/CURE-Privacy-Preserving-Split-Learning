// cure-client: Client-side component for split learning with HE
package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn"
	"cure_lib/nn/layers"
	"cure_lib/split"
	"cure_lib/tensor"
	"cure_lib/utils"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

var (
	logN         = flag.Int("logN", 13, "Ring dimension log2")
	modelType    = flag.String("model", "mnist", "Model type: mnist, bcw")
	epochs       = flag.Int("epochs", 5, "Training epochs")
	samples      = flag.Int("samples", 10, "Synthetic samples")
	learningRate = flag.Float64("lr", 0.01, "Learning rate")
	verbose      = flag.Bool("verbose", false, "Verbose output")
	seed         = flag.Int64("seed", 42, "Random seed")
)

func main() {
	flag.Parse()
	utils.Verbose = *verbose
	rand.Seed(*seed)

	log("CURE Client starting (logN=%d, model=%s)", *logN, *modelType)

	// Initialize HE
	heCtx := ckkswrapper.NewHeContextWithLogN(*logN)

	// Build client layers (plaintext)
	var clientLayers []nn.Module
	var inputDim, hiddenDim, outputDim int

	switch *modelType {
	case "mnist":
		inputDim, hiddenDim, outputDim = 784, 128, 10
		clientLayers = []nn.Module{
			layers.NewLinear(128, 32, false, nil),
			mustAct("ReLU3", false, nil),
			layers.NewLinear(32, 10, false, nil),
		}
	case "bcw":
		inputDim, hiddenDim, outputDim = 64, 32, 10
		clientLayers = []nn.Module{
			layers.NewLinear(32, 16, false, nil),
			mustAct("ReLU3", false, nil),
			layers.NewLinear(16, 10, false, nil),
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown model: %s\n", *modelType)
		os.Exit(1)
	}

	// Init weights
	for _, layer := range clientLayers {
		if lin, ok := layer.(*layers.Linear); ok {
			scale := math.Sqrt(2.0 / float64(lin.W.Shape[0]+lin.W.Shape[1]))
			for i := range lin.W.Data {
				lin.W.Data[i] = rand.NormFloat64() * scale
			}
		}
	}

	// Generate data
	inputs, labels := generateData(inputDim, outputDim, *samples)

	// Protocol
	protocol := split.NewProtocol(os.Stdin, os.Stdout)

	log("Starting training...")
	start := time.Now()

	for epoch := 0; epoch < *epochs; epoch++ {
		epochLoss := 0.0

		for i := 0; i < len(inputs); i++ {
			// Encrypt input
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			inputC := make([]complex128, heCtx.Params.MaxSlots())
			for j := 0; j < inputDim; j++ {
				inputC[j] = complex(inputs[i].Data[j], 0)
			}
			heCtx.Encoder.Encode(inputC, pt)
			ct, _ := heCtx.Encryptor.EncryptNew(pt)

			// Send to server
			ctBytes, _ := ct.MarshalBinary()
			protocol.SendForward(i, ctBytes, ct.Level(), 0)

			// Receive from server
			resp, err := protocol.ReceiveForward()
			if err == io.EOF {
				break
			}
			if err != nil {
				log("Error: %v", err)
				continue
			}

			// Decrypt
			ctOut := new(rlwe.Ciphertext)
			ctOut.UnmarshalBinary(resp.Ciphertext)
			ptOut := heCtx.Decryptor.DecryptNew(ctOut)
			decoded := make([]complex128, heCtx.Params.MaxSlots())
			heCtx.Encoder.Decode(ptOut, decoded)

			// Convert to tensor
			hidden := tensor.New(hiddenDim)
			for j := 0; j < hiddenDim; j++ {
				hidden.Data[j] = real(decoded[j])
			}

			// Client forward
			var out interface{} = hidden
			for _, layer := range clientLayers {
				out, _ = layer.Forward(out)
			}

			// Loss
			outT := out.(*tensor.Tensor)
			probs := nn.Softmax(outT)
			loss := 0.0
			for j := 0; j < outputDim; j++ {
				if labels[i].Data[j] > 0 {
					p := probs.Data[j]
					if p < 1e-10 {
						p = 1e-10
					}
					loss -= labels[i].Data[j] * math.Log(p)
				}
			}
			epochLoss += loss

			// Backward
			grad := tensor.New(outputDim)
			for j := 0; j < outputDim; j++ {
				grad.Data[j] = probs.Data[j] - labels[i].Data[j]
			}
			var gradOut interface{} = grad
			for j := len(clientLayers) - 1; j >= 0; j-- {
				gradOut, _ = clientLayers[j].Backward(gradOut)
			}
		}

		log("Epoch %d/%d | Loss: %.6f", epoch+1, *epochs, epochLoss/float64(len(inputs)))
	}

	protocol.SendDone()
	log("Training complete (%.2fs)", time.Since(start).Seconds())
}

func mustAct(name string, encrypted bool, heCtx *ckkswrapper.HeContext) nn.Module {
	act, _ := layers.NewActivation(name, encrypted, heCtx)
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

func log(format string, args ...interface{}) {
	if *verbose {
		fmt.Fprintf(os.Stderr, "[CLIENT] "+format+"\n", args...)
	}
}
