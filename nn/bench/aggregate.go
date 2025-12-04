package bench

import (
	"cure_lib/nn/layers"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"cure_lib/core/ckkswrapper"
	"sync"
)

type Point struct {
	Net, CutIdx   string
	LogN, Cores   int
	Fwd, Bwd, Upd time.Duration
	Rot, Mul      int
}

// RunPoint runs the benchmark for a given configuration and returns a Point
func RunPoint(net BuiltNet, cut int, logN, cores int) (Point, error) {
	runtime.GOMAXPROCS(cores)
	// heCtx := ckkswrapper.NewHeContextWithLogN(logN) // Not needed here

	L := len(net.Layers)
	layerHE := make([]struct {
		fwd, bwd, upd time.Duration
		rot, mul      int
	}, L)
	layerPlain := make([]struct {
		fwd, bwd, upd time.Duration
		rot, mul      int
	}, L)

	slots := 0 // TODO: set slots as needed for each model
	numRuns := 3

	// Time each layer in both HE and plain modes
	if net.Name == "audio1d" {
		isAudio1DModel = true
	}
	if net.Name == "resnet" {
		isResNetModel = true
	}
	for i, layer := range net.Layers {
		// Set encrypted true/false for timing
		if setEnc, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
			setEnc.EnableEncrypted(true)
		}
		fwd, bwd, upd, rot, mul := TimeLayer(layer, slots, numRuns)
		layerHE[i] = struct {
			fwd, bwd, upd time.Duration
			rot, mul      int
		}{fwd, bwd, upd, rot, mul}

		if setEnc, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
			setEnc.EnableEncrypted(false)
		}
		fwd, bwd, upd, rot, mul = TimeLayer(layer, slots, numRuns)
		layerPlain[i] = struct {
			fwd, bwd, upd time.Duration
			rot, mul      int
		}{fwd, bwd, upd, rot, mul}
	}
	isAudio1DModel = false
	isResNetModel = false

	// --- MICROBENCHMARKING LOGIC ---
	// Helper to identify layer type and parameters as a string key
	layerKey := func(layer interface{}) string {
		switch l := layer.(type) {
		case *layers.Linear:
			return fmt.Sprintf("Linear(%d,%d)", l.W.Shape[1], l.W.Shape[0])
		case *layers.Activation:
			return fmt.Sprintf("Activation(%s,deg=%d)", l.Poly().Name, l.Poly().Degree)
		default:
			return fmt.Sprintf("%T", layer)
		}
	}

	// Microbenchmark struct
	// Holds timing for fwd, bwd, upd in HE and Plain
	type Microbenchmark struct {
		HEFwd, HEBwd, HEUpd          time.Duration
		PlainFwd, PlainBwd, PlainUpd time.Duration
	}

	// Build a microbenchmark dictionary for each distinct layer type/parameter set (Linear, Activation, etc.).
	layerMicrobenchmarks := make(map[string]Microbenchmark)
	for i := 0; i < L; i++ {
		key := layerKey(net.Layers[i])
		if _, exists := layerMicrobenchmarks[key]; !exists {
			layerMicrobenchmarks[key] = Microbenchmark{}
		}

		mb := layerMicrobenchmarks[key]
		if i < cut {
			mb.PlainFwd += layerPlain[i].fwd
			mb.PlainBwd += layerPlain[i].bwd
			mb.PlainUpd += layerPlain[i].upd
		} else {
			mb.HEFwd += layerHE[i].fwd
			mb.HEBwd += layerHE[i].bwd
			mb.HEUpd += layerHE[i].upd
		}
		layerMicrobenchmarks[key] = mb
	}

	// Print the microbenchmarks and the aggregation for cuts 0–5 for mnistfc as a demonstration.
	if net.Name == "mnistfc" {
		fmt.Printf("Microbenchmarks for mnistfc:\n")
		for key, mb := range layerMicrobenchmarks {
			fmt.Printf("Layer: %s\n", key)
			fmt.Printf("HE Fwd: %v, Bwd: %v, Upd: %v\n", mb.HEFwd, mb.HEBwd, mb.HEUpd)
			fmt.Printf("Plain Fwd: %v, Bwd: %v, Upd: %v\n", mb.PlainFwd, mb.PlainBwd, mb.PlainUpd)
			fmt.Printf("Total Fwd: %v, Bwd: %v, Upd: %v\n", mb.HEFwd+mb.PlainFwd, mb.HEBwd+mb.PlainBwd, mb.HEUpd+mb.PlainUpd)
			fmt.Println("---")
		}
	}

	// MICROBENCHMARK TABLE
	fmt.Println("\nMicrobenchmark Table (per layer, per op, HE and Plain):")
	fmt.Printf("%-30s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n", "Layer", "HE Fwd", "HE Bwd", "HE Upd", "Plain Fwd", "Plain Bwd", "Plain Upd")
	for i, layer := range net.Layers {
		key := layerKey(layer)
		fmt.Printf("%-30s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n",
			key,
			layerHE[i].fwd, layerHE[i].bwd, layerHE[i].upd,
			layerPlain[i].fwd, layerPlain[i].bwd, layerPlain[i].upd)
	}

	// AGGREGATION PER CUT
	fmt.Println("\nAggregation Results by Cut:")
	for cut := 0; cut <= L; cut++ {
		var sumFwd, sumBwd, sumUpd time.Duration
		var sumRot, sumMul int
		arch := make([]string, L)
		for i := 0; i < L; i++ {
			if i < cut {
				sumFwd += layerHE[i].fwd
				sumBwd += layerHE[i].bwd
				sumUpd += layerHE[i].upd
				sumRot += layerHE[i].rot
				sumMul += layerHE[i].mul
				arch[i] = "HE"
			} else {
				sumFwd += layerPlain[i].fwd
				sumBwd += layerPlain[i].bwd
				sumUpd += layerPlain[i].upd
				sumRot += layerPlain[i].rot
				sumMul += layerPlain[i].mul
				arch[i] = "Plain"
			}
		}
		fmt.Printf("Cut %d: [%s] | Fwd: %v | Bwd: %v | Upd: %v | Rot: %d | Mul: %d\n",
			cut, strings.Join(arch, ", "), sumFwd, sumBwd, sumUpd, sumRot, sumMul)
	}

	// Aggregate timings for this cut
	var sumFwd, sumBwd, sumUpd time.Duration
	var sumRot, sumMul int
	for i := 0; i < L; i++ {
		if i < cut {
			sumFwd += layerPlain[i].fwd
			sumBwd += layerPlain[i].bwd
			sumUpd += layerPlain[i].upd
			sumRot += layerPlain[i].rot
			sumMul += layerPlain[i].mul
		} else {
			sumFwd += layerHE[i].fwd
			sumBwd += layerHE[i].bwd
			sumUpd += layerHE[i].upd
			sumRot += layerHE[i].rot
			sumMul += layerHE[i].mul
		}
	}

	pt := Point{
		Net:    net.Name,
		CutIdx: string(rune(cut)),
		LogN:   logN,
		Cores:  cores,
		Fwd:    sumFwd,
		Bwd:    sumBwd,
		Upd:    sumUpd,
		Rot:    sumRot,
		Mul:    sumMul,
	}

	// After running the benchmark for the current architecture and CKKS parameters, write the microbenchmark table to a .txt file
	outputFile := fmt.Sprintf("microbench_%s_logN%d.txt", net.Name, logN)
	f, err := os.Create(outputFile)
	if err != nil {
		fmt.Printf("Failed to create microbenchmark output file: %v\n", err)
	} else {
		defer f.Close()
		fmt.Fprintf(f, "Microbenchmark Table for %s (logN=%d)\n", net.Name, logN)
		fmt.Fprintf(f, "%-30s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n", "Layer", "HE Fwd", "HE Bwd", "HE Upd", "Plain Fwd", "Plain Bwd", "Plain Upd")
		for _, layer := range net.Layers {
			key := layerKey(layer)
			mb := layerMicrobenchmarks[key]
			fmt.Fprintf(f, "%-30s | %9s ns | %9s ns | %9s ns | %9s µs | %9s µs | %9s µs\n",
				key,
				mb.HEFwd.String(), mb.HEBwd.String(), mb.HEUpd.String(),
				toMicro(mb.PlainFwd), toMicro(mb.PlainBwd), toMicro(mb.PlainUpd))
		}
	}

	return pt, nil
}

// Helper to summarize CKKS parameters
func CKKSParamsSummary(heCtx *ckkswrapper.HeContext) string {
	if heCtx == nil {
		return ""
	}
	params := heCtx.Params
	return fmt.Sprintf("logN=%d,logQ=%v,logP=%v", params.LogN(), params.LogQ(), params.LogP())
}

// RunLayerBenchmarks runs the per-layer, per-mode, per-run microbenchmarking protocol as described in AGENTS/23.md
func RunLayerBenchmarks(net BuiltNet, logN, cores int, heCtx *ckkswrapper.HeContext, csvFile *os.File) error {
	runtime.GOMAXPROCS(cores)
	slots := 0 // TODO: set slots as needed for each model
	numRuns := 3

	layerKey := func(layer interface{}) string {
		switch l := layer.(type) {
		case *layers.Linear:
			return fmt.Sprintf("Linear(%d,%d)", l.W.Shape[1], l.W.Shape[0])
		case *layers.Activation:
			return fmt.Sprintf("Activation(%s,deg=%d)", l.Poly().Name, l.Poly().Degree)
		default:
			return fmt.Sprintf("%T", layer)
		}
	}

	ckksSummary := CKKSParamsSummary(heCtx)

	for i, layer := range net.Layers {
		// Skip MaxPool1D and Flatten layers
		switch layer.(type) {
		case *layers.MaxPool1D, *layers.Flatten:
			continue
		}
		for run := 1; run <= numRuns; run++ {
			// HE mode
			if setEnc, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
				setEnc.EnableEncrypted(true)
			}
			fwdHE, bwdHE, updHE, _, _ := TimeLayer(layer, slots, 1)
			fmt.Fprintf(csvFile, "%s,%d,%s,HE,%d,%.3f,%.3f,%.3f,,,,%s\n",
				net.Name, i, layerKey(layer), run,
				float64(fwdHE.Nanoseconds())/1000.0,
				float64(bwdHE.Nanoseconds())/1000.0,
				float64(updHE.Nanoseconds())/1000.0,
				ckksSummary)

			// Plain mode
			if setEnc, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
				setEnc.EnableEncrypted(false)
			}
			fwdP, bwdP, updP, _, _ := TimeLayer(layer, slots, 1)
			fmt.Fprintf(csvFile, "%s,%d,%s,Plain,%d,,,,%.3f,%.3f,%.3f,%s\n",
				net.Name, i, layerKey(layer), run,
				float64(fwdP.Nanoseconds())/1000.0,
				float64(bwdP.Nanoseconds())/1000.0,
				float64(updP.Nanoseconds())/1000.0,
				ckksSummary)
		}
	}
	return nil
}

// RunLayerBenchmarksParallel runs numIters iterations per (layer, mode) in parallel, averages timings, and writes a single row per (model, layer, mode) to the CSV.
func RunLayerBenchmarksParallel(net BuiltNet, logN, cores int, heCtx *ckkswrapper.HeContext, csvFile *os.File, numIters int) error {
	runtime.GOMAXPROCS(cores)
	slots := 0 // TODO: set slots as needed for each model

	layerKey := func(layer interface{}) string {
		switch l := layer.(type) {
		case *layers.Linear:
			return fmt.Sprintf("Linear(%d,%d)", l.W.Shape[1], l.W.Shape[0])
		case *layers.Activation:
			return fmt.Sprintf("Activation(%s,deg=%d)", l.Poly().Name, l.Poly().Degree)
		default:
			return fmt.Sprintf("%T", layer)
		}
	}

	ckksSummary := CKKSParamsSummary(heCtx)

	for i, layer := range net.Layers {
		// Skip MaxPool1D and Flatten layers
		switch layer.(type) {
		case *layers.MaxPool1D, *layers.Flatten:
			continue
		}
		for _, mode := range []string{"HE", "Plain"} {
			if setEnc, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
				setEnc.EnableEncrypted(mode == "HE")
			}
			var totalFwd, totalBwd, totalUpd int64
			var wg sync.WaitGroup
			mu := &sync.Mutex{}
			itersPerWorker := 10
			nWorkers := numIters / itersPerWorker
			if numIters%itersPerWorker != 0 {
				nWorkers++
			}
			wg.Add(nWorkers)
			for w := 0; w < nWorkers; w++ {
				go func() {
					defer wg.Done()
					localFwd, localBwd, localUpd := int64(0), int64(0), int64(0)
					for j := 0; j < itersPerWorker; j++ {
						fwd, bwd, upd, _, _ := TimeLayer(layer, slots, 1)
						localFwd += fwd.Nanoseconds()
						localBwd += bwd.Nanoseconds()
						localUpd += upd.Nanoseconds()
					}
					mu.Lock()
					totalFwd += localFwd
					totalBwd += localBwd
					totalUpd += localUpd
					mu.Unlock()
				}()
			}
			wg.Wait()
			iters := int64(numIters)
			avgFwd := float64(totalFwd) / float64(iters) / 1000.0
			avgBwd := float64(totalBwd) / float64(iters) / 1000.0
			avgUpd := float64(totalUpd) / float64(iters) / 1000.0
			encVal := "false"
			if mode == "HE" {
				encVal = "true"
			}
			fmt.Fprintf(csvFile, "%s,%d,%s,%s,%d,%.3f,%.3f,%.3f,%s\n",
				net.Name, i, layerKey(layer), encVal, cores, avgFwd, avgBwd, avgUpd, ckksSummary)
		}
	}
	return nil
}

// Helper to convert time.Duration to microseconds string
func toMicro(d time.Duration) string {
	return fmt.Sprintf("%.3f", float64(d.Nanoseconds())/1000.0)
}
