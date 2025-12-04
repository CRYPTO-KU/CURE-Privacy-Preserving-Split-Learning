package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn"
	benchpkg "cure_lib/nn/bench"
	"cure_lib/nn/layers"
)

// parseCSVInts parses a comma-separated list of integers
func parseCSVInts(s string) ([]int, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid int %q: %w", p, err)
		}
		out = append(out, v)
	}
	return out, nil
}

// layerNameForCSV matches the naming used in existing bench CSVs
func layerNameForCSV(m interface{}) string {
	switch l := m.(type) {
	case *layers.Linear:
		inDim := l.W.Shape[1]
		outDim := l.W.Shape[0]
		return fmt.Sprintf("Linear_%d_%d", inDim, outDim)
	case *layers.Activation:
		return fmt.Sprintf("Activation_%s", l.Poly().Name)
	case *layers.MaxPool1D:
		return fmt.Sprintf("MaxPool1D_%d", l.Window)
	case *layers.Flatten:
		return "Flatten"
	default:
		// Fallback to type name for non-FC layers
		return fmt.Sprintf("%T", m)
	}
}

// buildNetByName constructs the network by string name
func buildNetByName(name string, heCtx *ckkswrapper.HeContext) (benchpkg.BuiltNet, error) {
	switch strings.ToLower(name) {
	case "mnistfc":
		return benchpkg.BuildMNISTFC(heCtx, true), nil
	case "bcwfc":
		return benchpkg.BuildBCWFC(heCtx, true), nil
	case "lenet":
		return benchpkg.BuildLeNet(heCtx, true), nil
	case "audio1d":
		return benchpkg.BuildAudio1D(heCtx, true), nil
	case "resnet":
		return benchpkg.BuildResNetBlock(heCtx, true), nil
	default:
		return benchpkg.BuiltNet{}, fmt.Errorf("unknown model %q", name)
	}
}

// measureLayer averages times over iters with warmup runs
func measureLayer(layer nn.Module, slots int, iters int, warmup int) (avgFwd, avgBwd, avgUpd time.Duration) {
	// Warmup without recording
	for i := 0; i < warmup; i++ {
		benchpkg.TimeLayer(layer, slots, 1)
	}

	totalFwd := time.Duration(0)
	totalBwd := time.Duration(0)
	totalUpd := time.Duration(0)
	for i := 0; i < iters; i++ {
		f, b, u, _, _ := benchpkg.TimeLayer(layer, slots, 1)
		totalFwd += f
		totalBwd += b
		totalUpd += u
	}
	denom := time.Duration(iters)
	if iters == 0 {
		denom = 1
	}
	return totalFwd / denom, totalBwd / denom, totalUpd / denom
}

func main() {
	var modelsCSV string
	var logNsCSV string
	var coresCSV string
	var outPath string
	var iters int
	var warmup int
	var includePlain bool
	var casesFile string

	flag.StringVar(&modelsCSV, "models", "mnistfc,bcwfc", "Comma-separated list of models to run (e.g., mnistfc,bcwfc,lenet)")
	flag.StringVar(&logNsCSV, "logNs", "13", "Comma-separated list of logN values (e.g., 13,14,15)")
	flag.StringVar(&coresCSV, "cores", "32,40", "Comma-separated list of core counts to run (e.g., 32,40)")
	flag.StringVar(&outPath, "out", "bench_rerun_results.csv", "Output CSV path")
	flag.IntVar(&iters, "iters", 100, "Iterations per (layer, mode) for averaging")
	flag.IntVar(&warmup, "warmup", 5, "Warmup runs per (layer, mode) before timing")
	flag.BoolVar(&includePlain, "include-plain", true, "Also record Plain mode timings (logN will be '-')")
	flag.StringVar(&casesFile, "cases-file", "", "Optional CSV file with columns including model,logN,(num_cores|cores)[,layer] to restrict runs to only these cases")
	flag.Parse()

	// Optional: parse cases-file to restrict runs
	type rerunCase struct {
		model  string
		logN   int
		cores  int
		layers map[string]bool // if empty: run all layers; if non-empty: only run those layer names
	}
	cases := []rerunCase{}
	if casesFile != "" {
		cf, err := os.Open(casesFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to open cases-file: %v\n", err)
			os.Exit(2)
		}
		defer cf.Close()
		r := csv.NewReader(cf)
		r.FieldsPerRecord = -1
		head, err := r.Read()
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to read cases-file header: %v\n", err)
			os.Exit(2)
		}
		// build column index
		col := map[string]int{}
		for i, h := range head {
			col[strings.ToLower(strings.TrimSpace(h))] = i
		}
		// helper to fetch field by any of acceptable names
		get := func(rec []string, names ...string) string {
			for _, n := range names {
				if idx, ok := col[strings.ToLower(n)]; ok && idx < len(rec) {
					return strings.TrimSpace(rec[idx])
				}
			}
			return ""
		}
		// map key: model|logN|cores
		acc := map[string]rerunCase{}
		for {
			rec, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed to read cases-file row: %v\n", err)
				os.Exit(2)
			}
			model := get(rec, "model")
			if model == "" {
				continue
			}
			logNS := get(rec, "logn", "log_n")
			coresS := get(rec, "num_cores", "cores")
			if logNS == "" || coresS == "" {
				continue
			}
			logN, err1 := strconv.Atoi(logNS)
			cores, err2 := strconv.Atoi(coresS)
			if err1 != nil || err2 != nil {
				continue
			}
			layer := get(rec, "layer")
			key := fmt.Sprintf("%s|%d|%d", model, logN, cores)
			c := acc[key]
			if c.model == "" {
				c = rerunCase{model: model, logN: logN, cores: cores, layers: map[string]bool{}}
			}
			if layer != "" {
				c.layers[layer] = true
			}
			acc[key] = c
		}
		for _, v := range acc {
			cases = append(cases, v)
		}
	}

	models := []string{}
	for _, m := range strings.Split(modelsCSV, ",") {
		ms := strings.TrimSpace(m)
		if ms != "" {
			models = append(models, ms)
		}
	}
	logNs, err := parseCSVInts(logNsCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid logNs: %v\n", err)
		os.Exit(2)
	}
	coresList, err := parseCSVInts(coresCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid cores: %v\n", err)
		os.Exit(2)
	}

	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create output CSV: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	// Header consistent with existing analysis
	w.Write([]string{"model", "layer", "mode", "logN", "forward_time", "backward_time", "update_time", "num_cores"})

	slots := 0

	if len(cases) > 0 {
		// Restricted mode: run only cases from cases-file
		for _, c := range cases {
			heCtx := ckkswrapper.NewHeContextWithLogN(c.logN)
			net, err := buildNetByName(c.model, heCtx)
			if err != nil {
				fmt.Fprintf(os.Stderr, "skip model %s: %v\n", c.model, err)
				continue
			}
			runtime.GOMAXPROCS(c.cores)
			for _, layer := range net.Layers {
				// Skip always-plaintext layers
				switch layer.(type) {
				case *layers.MaxPool1D, *layers.Flatten:
					continue
				}
				lname := layerNameForCSV(layer)
				if len(c.layers) > 0 {
					if _, ok := c.layers[lname]; !ok {
						continue
					}
				}
				if setter, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
					setter.EnableEncrypted(true)
				}
				fwdHE, bwdHE, updHE := measureLayer(layer, slots, iters, warmup)
				heRow := []string{c.model, lname, "HE", strconv.Itoa(c.logN),
					fmt.Sprintf("%.6f", float64(fwdHE.Nanoseconds())/1e9),
					fmt.Sprintf("%.6f", float64(bwdHE.Nanoseconds())/1e9),
					fmt.Sprintf("%.6f", float64(updHE.Nanoseconds())/1e9),
					strconv.Itoa(c.cores)}
				w.Write(heRow)
				if includePlain {
					if setter, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
						setter.EnableEncrypted(false)
					}
					fwdP, bwdP, updP := measureLayer(layer, slots, iters, warmup)
					plainRow := []string{c.model, lname, "Plain", "-",
						fmt.Sprintf("%.6f", float64(fwdP.Nanoseconds())/1e9),
						fmt.Sprintf("%.6f", float64(bwdP.Nanoseconds())/1e9),
						fmt.Sprintf("%.6f", float64(updP.Nanoseconds())/1e9),
						strconv.Itoa(c.cores)}
					w.Write(plainRow)
				}
			}
			w.Flush()
		}
	} else {
		// Default mode: Cartesian product of flags
		for _, logN := range logNs {
			heCtx := ckkswrapper.NewHeContextWithLogN(logN)
			for _, modelName := range models {
				net, err := buildNetByName(modelName, heCtx)
				if err != nil {
					fmt.Fprintf(os.Stderr, "skip model %s: %v\n", modelName, err)
					continue
				}
				for _, cores := range coresList {
					runtime.GOMAXPROCS(cores)
					for _, layer := range net.Layers {
						switch layer.(type) {
						case *layers.MaxPool1D, *layers.Flatten:
							continue
						}
						if setter, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
							setter.EnableEncrypted(true)
						}
						fwdHE, bwdHE, updHE := measureLayer(layer, slots, iters, warmup)
						heRow := []string{
							modelName,
							layerNameForCSV(layer),
							"HE",
							strconv.Itoa(logN),
							fmt.Sprintf("%.6f", float64(fwdHE.Nanoseconds())/1e9),
							fmt.Sprintf("%.6f", float64(bwdHE.Nanoseconds())/1e9),
							fmt.Sprintf("%.6f", float64(updHE.Nanoseconds())/1e9),
							strconv.Itoa(cores),
						}
						w.Write(heRow)
						if includePlain {
							if setter, ok := layer.(interface{ EnableEncrypted(bool) }); ok {
								setter.EnableEncrypted(false)
							}
							fwdP, bwdP, updP := measureLayer(layer, slots, iters, warmup)
							plainRow := []string{
								modelName,
								layerNameForCSV(layer),
								"Plain",
								"-",
								fmt.Sprintf("%.6f", float64(fwdP.Nanoseconds())/1e9),
								fmt.Sprintf("%.6f", float64(bwdP.Nanoseconds())/1e9),
								fmt.Sprintf("%.6f", float64(updP.Nanoseconds())/1e9),
								strconv.Itoa(cores),
							}
							w.Write(plainRow)
						}
					}
					w.Flush()
				}
			}
		}
	}

	fmt.Printf("Wrote results to %s\n", outPath)
}
