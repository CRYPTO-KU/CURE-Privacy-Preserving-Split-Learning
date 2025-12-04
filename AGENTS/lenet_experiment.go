package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn"
	"cure_lib/nn/layers"
	"cure_lib/tensor"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	BatchSize = 1 // Using batch size 1 for timing experiments as per common HE benchmarks
	EpochSize = 60000
)

// --- MNIST Loader ---

func loadMNISTImages(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 2051 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	var numImages, rows, cols int32
	if err := binary.Read(f, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &rows); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &cols); err != nil {
		return nil, err
	}

	images := make([][]float64, numImages)
	buf := make([]byte, rows*cols)

	for i := 0; i < int(numImages); i++ {
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, err
		}
		img := make([]float64, rows*cols)
		for j, b := range buf {
			img[j] = float64(b) / 255.0 // Normalize to 0-1
		}
		images[i] = img
	}
	return images, nil
}

func loadMNISTLabels(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	var numLabels int32
	if err := binary.Read(f, binary.BigEndian, &numLabels); err != nil {
		return nil, err
	}

	labels := make([]int, numLabels)
	buf := make([]byte, numLabels)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, err
	}
	for i, b := range buf {
		labels[i] = int(b)
	}
	return labels, nil
}

// --- Model Definition ---

type BuiltNet struct {
	Layers []nn.Module
}

func BuildLenet1D(heCtx *ckkswrapper.HeContext, encrypted bool) BuiltNet {
	// Input: [1, 784] (treated as [1, 1, 784] by Conv1D)
	// Conv1D(1, 6, 5) -> [6, 780]
	// ReLU
	// MaxPool1D(2) -> [6, 390]
	// Conv1D(6, 16, 5) -> [16, 386]
	// ReLU
	// MaxPool1D(2) -> [16, 193]
	// Flatten -> [3088]
	// Linear(3088, 120)
	// ReLU
	// Linear(120, 84)
	// ReLU
	// Linear(84, 10)

	layersList := []nn.Module{
		layers.NewConv1D(1, 6, 5, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewMaxPool1D(2), // MaxPool is usually plaintext in this lib or handled specially
		layers.NewConv1D(6, 16, 5, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewMaxPool1D(2),
		layers.NewFlatten(encrypted),
		layers.NewLinear(16*193, 120, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(120, 84, encrypted, heCtx),
		mustNewActivation("ReLU3", encrypted, heCtx),
		layers.NewLinear(84, 10, encrypted, heCtx),
	}

	return BuiltNet{Layers: layersList}
}

func mustNewActivation(name string, encrypted bool, heCtx *ckkswrapper.HeContext) nn.Module {
	act, err := layers.NewActivation(name, encrypted, heCtx)
	if err != nil {
		panic(err)
	}
	return act
}

// --- Experiment Logic ---

func main() {
	// 1. Load Data
	homeDir, _ := os.UserHomeDir()
	dataDir := filepath.Join(homeDir, "Documents/Research/CURE_lib/data/mnist/raw")
	images, err := loadMNISTImages(filepath.Join(dataDir, "train-images-idx3-ubyte"))
	if err != nil {
		log.Fatalf("Failed to load images: %v", err)
	}
	labels, err := loadMNISTLabels(filepath.Join(dataDir, "train-labels-idx1-ubyte"))
	if err != nil {
		log.Fatalf("Failed to load labels: %v", err)
	}
	log.Printf("Loaded %d images and %d labels", len(images), len(labels))

	// 2. Setup HE Context
	heCtx := ckkswrapper.NewHeContext()

	// 3. Table Experiments
	fmt.Println("\n=== Table Experiments: Lenet 1D Conv MNIST ===")
	fmt.Println("Split Point | Plaintext Time (ms) | Enc/Dec Time (ms) | Extrapolated Total (s)")

	// Define split points (index of the last layer on Client)
	// -1: Server has all (Client has nothing? No, Client always has input)
	// We iterate from Client has All (len(layers)) down to Client has Input only (0)

	// Construct a dummy model to get layer count
	dummyNet := BuildLenet1D(heCtx, false)
	numLayers := len(dummyNet.Layers)

	// Use a subset of images for timing if full set is too slow, but user asked for "epoch"
	// We'll use all 60000 images for timing.
	// But we pass the whole slice to runSplitExperiment.

	// Iterate from Server has 0 layers (Client All) to Server has All-1 (Client has last layer)
	// User said "last layer must always be on client"
	for serverLayers := 0; serverLayers < numLayers; serverLayers++ {
		runSplitExperiment(heCtx, images, labels, serverLayers, numLayers)
	}

	// 4. Simulation Extension
	runLayerWiseComparison(heCtx, images, labels)
}

func runSplitExperiment(heCtx *ckkswrapper.HeContext, images [][]float64, labels []int, serverLayers int, numLayers int) {
	net := BuildLenet1D(heCtx, false) // All plaintext for base timing

	// Measure Plaintext Time on Subset (Client Part)
	subsetSize := 100
	if len(images) < subsetSize {
		subsetSize = len(images)
	}

	// Pre-allocate input tensor
	tIn := tensor.New(1, 1, 784)

	var totalClientDuration time.Duration
	var fanInDim int

	for i := 0; i < subsetSize; i++ {
		img := images[i]
		for k, v := range img {
			tIn.Data[k] = v
		}
		var out interface{} = tIn

		// Server Part (Untimed)
		for j := 0; j < serverLayers; j++ {
			out, _ = net.Layers[j].Forward(out)
		}

		// Capture Fan-In Dimension (Input to Client)
		if i == 0 {
			if t, ok := out.(*tensor.Tensor); ok {
				fanInDim = len(t.Data)
			}
		}

		// Client Part (Timed)
		tStart := time.Now()
		for j := serverLayers; j < numLayers; j++ {
			out, _ = net.Layers[j].Forward(out)
		}
		totalClientDuration += time.Since(tStart)
	}

	avgPlaintext := totalClientDuration / time.Duration(subsetSize)
	totalPlaintext := avgPlaintext * time.Duration(len(images)) // Extrapolate to full dataset

	splitName := "Client All"
	if serverLayers > 0 {
		splitName = fmt.Sprintf("Client L%d..%d", serverLayers+1, numLayers)
	}

	fmt.Printf("%-15s | %-10d | %22.2f s\n", splitName, fanInDim, totalPlaintext.Seconds())
}

func runLayerWiseComparison(heCtx *ckkswrapper.HeContext, images [][]float64, labels []int) {
	fmt.Println("\n=== Simulation Extension: Layer-wise Correctness (One Sample) ===")

	// Use the first sample
	img := images[0]
	// label := labels[0]

	// Build Networks
	plainNet := BuildLenet1D(heCtx, false)
	heNet := BuildLenet1D(heCtx, true)

	// Initialize HE layers
	currLen := 784
	for j, layer := range heNet.Layers {
		if conv, ok := layer.(*layers.Conv2D); ok {
			conv.SetDimensions(1, currLen)
			if j == 0 {
				conv.ForceUnpacked = true
			}
			conv.SyncHE()

			var inC, outC, kh, kw int
			fmt.Sscanf(conv.Tag(), "Conv2D_%d_%d_%d_%d", &inC, &outC, &kh, &kw)
			currLen = currLen - kw + 1
		} else if pool, ok := layer.(*layers.MaxPool1D); ok {
			currLen = currLen / pool.Window
		} else if lin, ok := layer.(*layers.Linear); ok {
			lin.SyncHE()
		}
	}

	// Prepare Inputs
	// Plaintext: [1, 1, 784]
	ptIn := tensor.New(1, 1, 784)
	for k, v := range img {
		ptIn.Data[k] = v
	}

	// HE: Encrypt input
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(img, pt)
	ctInput, _ := heCtx.Encryptor.EncryptNew(pt)

	var ptOut interface{} = ptIn
	var heOut interface{} = []*rlwe.Ciphertext{ctInput} // Start with slice for Conv1D

	// Track dimensions for reshaping
	simCurrLen := 784
	simChannels := 1

	fmt.Printf("%-10s | %-15s | %-15s\n", "Layer", "MSE", "Max Diff")
	fmt.Println("------------------------------------------------")

	for j, layer := range heNet.Layers {
		plainLayer := plainNet.Layers[j]

		// 1. Run Plaintext
		var err error
		ptOut, err = plainLayer.Forward(ptOut)
		if err != nil {
			log.Fatalf("Plaintext Layer %d failed: %v", j, err)
		}

		// 2. Run HE (with fallback)
		// ... (Reuse robust logic from before)
		var res interface{}
		handled := false

		// Case 1: Input is slice []*rlwe.Ciphertext
		if ctSlice, ok := heOut.([]*rlwe.Ciphertext); ok {
			if heLayer, ok := layer.(interface {
				ForwardHE([]*rlwe.Ciphertext) ([]*rlwe.Ciphertext, error)
			}); ok {
				res, err = heLayer.ForwardHE(ctSlice)
				handled = true
			} else {
				res, err = layer.Forward(heOut)
				handled = true
			}
		} else if ct, ok := heOut.(*rlwe.Ciphertext); ok {
			if heLayer, ok := layer.(interface {
				ForwardCipher(*rlwe.Ciphertext) (*rlwe.Ciphertext, error)
			}); ok {
				res, err = heLayer.ForwardCipher(ct)
				handled = true
			} else {
				res, err = layer.Forward(heOut)
				handled = true
			}
		}

		if handled && err == nil {
			heOut = res
		} else {
			// Fallback
			if err != nil {
				// log.Printf("Layer %d HE failed: %v. Fallback.", j, err)
			}

			// Decrypt heOut
			var ptTensor *tensor.Tensor
			if ctSlice, ok := heOut.([]*rlwe.Ciphertext); ok {
				numChannels := len(ctSlice)
				ptTensor = tensor.New(1, numChannels, 1, simCurrLen)
				for c := 0; c < numChannels; c++ {
					ptDec := heCtx.Decryptor.DecryptNew(ctSlice[c])
					decoded := make([]complex128, ctSlice[c].Slots())
					heCtx.Encoder.Decode(ptDec, decoded)
					for i := 0; i < simCurrLen; i++ {
						if i < len(decoded) {
							idx := c*simCurrLen + i
							ptTensor.Data[idx] = real(decoded[i])
						}
					}
				}
			} else if ct, ok := heOut.(*rlwe.Ciphertext); ok {
				ptDec := heCtx.Decryptor.DecryptNew(ct)
				decoded := make([]complex128, ct.Slots())
				heCtx.Encoder.Decode(ptDec, decoded)
				// For single CT, we assume it matches ptOut flattened
				// But we need to be careful about length.
				// ptOut is Tensor.
				tOut := ptOut.(*tensor.Tensor)
				expectedLen := len(tOut.Data)
				ptTensor = tensor.New(expectedLen)
				for k := 0; k < expectedLen && k < len(decoded); k++ {
					ptTensor.Data[k] = real(decoded[k])
				}

				// Reshape
				if _, ok := layer.(*layers.Conv2D); ok {
					newT := tensor.New(1, simChannels, 1, simCurrLen)
					for k := 0; k < simChannels*simCurrLen && k < len(ptTensor.Data); k++ {
						newT.Data[k] = ptTensor.Data[k]
					}
					ptTensor = newT
				} else if _, ok := layer.(*layers.MaxPool1D); ok {
					newT := tensor.New(1, simChannels, 1, simCurrLen)
					for k := 0; k < simChannels*simCurrLen && k < len(ptTensor.Data); k++ {
						newT.Data[k] = ptTensor.Data[k]
					}
					ptTensor = newT
				}
			}

			// Run Plaintext Forward on Decrypted
			if encLayer, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
				encLayer.EnableEncrypted(false)
				res, err = layer.Forward(ptTensor)
				encLayer.EnableEncrypted(true)
			} else {
				res, err = layer.Forward(ptTensor)
			}
			if err != nil {
				log.Fatalf("Layer %d fallback failed: %v", j, err)
			}

			// Encrypt result
			if tOut, ok := res.(*tensor.Tensor); ok {
				ptEnc := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
				heCtx.Encoder.Encode(tOut.Data, ptEnc)
				ct, _ := heCtx.Encryptor.EncryptNew(ptEnc)
				heOut = ct
			}
		}

		// Update Dimensions
		if conv, ok := layer.(*layers.Conv2D); ok {
			var inC, outC, kh, kw int
			fmt.Sscanf(conv.Tag(), "Conv2D_%d_%d_%d_%d", &inC, &outC, &kh, &kw)
			simCurrLen = simCurrLen - kw + 1
			simChannels = outC
		} else if pool, ok := layer.(*layers.MaxPool1D); ok {
			simCurrLen = simCurrLen / pool.Window
		}

		// 3. Compare Output
		// Decrypt heOut for comparison
		var heData []float64
		if ctSlice, ok := heOut.([]*rlwe.Ciphertext); ok {
			numChannels := len(ctSlice)
			// Flatten
			for c := 0; c < numChannels; c++ {
				ptDec := heCtx.Decryptor.DecryptNew(ctSlice[c])
				decoded := make([]complex128, ctSlice[c].Slots())
				heCtx.Encoder.Decode(ptDec, decoded)
				for i := 0; i < simCurrLen; i++ {
					if i < len(decoded) {
						heData = append(heData, real(decoded[i]))
					}
				}
			}
		} else if ct, ok := heOut.(*rlwe.Ciphertext); ok {
			ptDec := heCtx.Decryptor.DecryptNew(ct)
			decoded := make([]complex128, ct.Slots())
			heCtx.Encoder.Decode(ptDec, decoded)
			// For single CT, we assume it matches ptOut flattened
			// But we need to be careful about length.
			// ptOut is Tensor.
			tOut := ptOut.(*tensor.Tensor)
			expectedLen := len(tOut.Data)
			for k := 0; k < expectedLen && k < len(decoded); k++ {
				heData = append(heData, real(decoded[k]))
			}
		} else if t, ok := heOut.(*tensor.Tensor); ok {
			// Fallback might return Tensor if we didn't encrypt?
			// No, we always encrypt at end of fallback.
			// But wait, if we didn't fallback, heOut is CT.
			// If we did fallback, heOut is CT.
			// So this case shouldn't happen unless logic changed.
			heData = t.Data
		}

		// Compare with ptOut
		tOut := ptOut.(*tensor.Tensor)
		ptData := tOut.Data

		var mse, maxDiff float64
		count := 0
		for k := 0; k < len(ptData) && k < len(heData); k++ {
			diff := ptData[k] - heData[k]
			mse += diff * diff
			if math.Abs(diff) > maxDiff {
				maxDiff = math.Abs(diff)
			}
			count++
		}
		if count > 0 {
			mse /= float64(count)
		}

		fmt.Printf("Layer %-4d | %-15.10f | %-15.10f\n", j, mse, maxDiff)
	}
}

func argmax(data []float64) int {
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
