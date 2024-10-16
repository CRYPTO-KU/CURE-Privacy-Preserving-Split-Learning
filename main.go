package main

import (
	"cure/m"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

var flagShort = flag.Bool("short", false, "run the example with a smaller and insecure ring degree.")

func main() {

	// params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	// ==============================
	// === 1) RESIDUAL PARAMETERS ===
	// ==============================
	LogN := 14
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN: LogN,
		Q: []uint64{0x200000008001, 0x400018001, // 45 + 9 x 34
			0x3fffd0001, 0x400060001,
			0x400068001, 0x3fff90001,
			0x400080001, 0x4000a8001,
			0x400108001, 0x3ffeb8001},
		P:               []uint64{0x7fffffd8001, 0x7fffffc8001}, // 43, 43
		LogDefaultScale: 40,                                     // Log2 of the scale
	})
	if err != nil {
		panic(err)
	}
	// //==========================================
	// //=== 2) BOOTSTRAPPING PARAMETERSLITERAL ===
	// //==========================================
	// btpParametersLit := bootstrapping.ParametersLiteral{
	// 	LogP: []int{43, 43, 43},
	// }

	// //===================================
	// //=== 3) BOOTSTRAPPING PARAMETERS ===
	// //===================================
	// btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	// if err != nil {
	// 	panic(err)
	// }

	// if *flagShort {
	// 	// Corrects the message ratio Q0/|m(X)| to take into account the smaller number of slots and keep the same precision
	// 	btpParams.Mod1ParametersLiteral.LogMessageRatio += 16 - params.LogN()
	// }

	// // We print some information about the residual parameters.
	// fmt.Printf("Residual parameters: logN=%d, logSlots=%d, H=%d, sigma=%f, logQP=%f, levels=%d, scale=2^%d\n",
	// 	btpParams.ResidualParameters.LogN(),
	// 	btpParams.ResidualParameters.LogMaxSlots(),
	// 	btpParams.ResidualParameters.XsHammingWeight(),
	// 	btpParams.ResidualParameters.Xe(), params.LogQP(),
	// 	btpParams.ResidualParameters.MaxLevel(),
	// 	btpParams.ResidualParameters.LogDefaultScale())

	// // And some information about the bootstrapping parameters.
	// // We can notably check that the LogQP of the bootstrapping parameters is smaller than 1550, which ensures
	// // 128-bit of security as explained above.
	// fmt.Printf("Bootstrapping parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%f, logQP=%f, levels=%d, scale=2^%d\n",
	// 	btpParams.BootstrappingParameters.LogN(),
	// 	btpParams.BootstrappingParameters.LogMaxSlots(),
	// 	btpParams.BootstrappingParameters.XsHammingWeight(),
	// 	btpParams.EphemeralSecretWeight,
	// 	btpParams.BootstrappingParameters.Xe(),
	// 	btpParams.BootstrappingParameters.LogQP(),
	// 	btpParams.BootstrappingParameters.QCount(),
	// 	btpParams.BootstrappingParameters.LogDefaultScale())

	//===========================
	//=== 4) KEYGEN & ENCRYPT ===
	//===========================
	kgen := hefloat.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	upperlimit := 24
	rotations := make([]int, upperlimit)
	for i := 1; i <= upperlimit; i++ {
		rotations[i-1] = i * 128
	}
	encoder := hefloat.NewEncoder(params)
	// rotKey := kgen.GenRotationKeysForRotations(rotations, true, sk)
	encryptor := hefloat.NewEncryptor(params, pk)
	decryptor := hefloat.NewDecryptor(params, sk)
	// fmt.Println("Generating bootstrapping keys...")
	// // evk, _, err := btpParams.GenEvaluationKeys(sk)
	// // if err != nil {
	// // 	panic(err)
	// // }
	// fmt.Println("Done")
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := hefloat.NewEvaluator(params, evk)

	galEls := []uint64{
		params.GaloisElement(1),
		params.GaloisElement(2),
		params.GaloisElement(4),
		params.GaloisElement(8),
		params.GaloisElement(16),
		params.GaloisElement(32),
		params.GaloisElement(64),
		params.GaloisElement(128),
		params.GaloisElement(256),
		params.GaloisElement(512),
		params.GaloisElement(1024),
		params.GaloisElement(2048),
		params.GaloisElement(4096),
		params.GaloisElement(8192),
	}
	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))
	//========================
	//=== 5) BOOTSTRAPPING ===
	//========================
	// Instantiates the bootstrapper
	// var btpEvaluator *bootstrapping.Evaluator
	// if btpEvaluator, err = bootstrapping.NewEvaluator(btpParams, evk); err != nil {
	// 	panic(err)
	// }
	// Note: Assuming you have already created params, sk, and btpParams
	// Ensure that you have correct imports and the required parameters

	encryptionElements := m.EncryptionElements{
		Params:    params,
		Encoder:   encoder,
		Encryptor: encryptor,
		Evaluator: eval,
		Decryptor: decryptor,
		// btpEvaluator: btpEvaluator,
	}

	if len(os.Args) < 2 {
		fmt.Println("a command and dataset must be specified")
		os.Exit(1)
	}
	subCommand := os.Args[1]
	networkName := os.Args[2]

	switch subCommand {
	case "train":
		// seed rand with pseudo random values
		rand.NewSource(time.Now().UTC().UnixNano())

		// parse training flags
		trainFlags := flag.NewFlagSet("train", flag.ContinueOnError)
		flagNumInputs := trainFlags.Int("input", 784, "input controls the number of input nodes")
		flagNumHidden := trainFlags.String("hidden", "30", "output controls the number of hidden nodes (comma-separated)")
		flagNumOutput := trainFlags.Int("output", 10, "output controls the number of output nodes")
		flagNumLayers := trainFlags.Int("layers", 3, "layers controls the total number of layers to use (3 means one hidden)")
		flagNumEpochs := trainFlags.Int("epochs", 6, "number of epochs")
		flagActivator := trainFlags.String("activator", "sigmoid", "activator is the activation function to use (default is sigmoid)")
		flagLearningRate := trainFlags.Float64("rate", .05, "rate is the learning rate")
		flagTargetLabels := trainFlags.String("labels", "0,1,2,3,4,5,6,7,8,9", "labels are name to call each output")
		flagBatchSize := trainFlags.Int("batch", 60, "batch size")
		flagCipherIndex := trainFlags.Int("cipherIndex", 2, "cipher index")
		flagENCRYPTED := trainFlags.Bool("ENCRYPTED", false, "encrypted bool")

		err := trainFlags.Parse(os.Args[3:])
		if err != nil {
			fmt.Printf("parsing train flags: %s\n", err.Error())
			os.Exit(1)
		}

		hiddenLayerNeurons, err := parseHiddenLayers(*flagNumHidden)
		if err != nil {
			fmt.Println("Error parsing hidden layers:", err)
			os.Exit(1)
		}

		if *flagNumLayers < 3 {
			fmt.Println("cannot have fewer than three layers")
			os.Exit(1)
		}

		activator, ok := m.ActivatorLookup[*flagActivator]
		if !ok {
			fmt.Println("invalid activator")
			os.Exit(1)
		}

		labelSplits := strings.Split(*flagTargetLabels, ",")
		if len(labelSplits) != *flagNumOutput {
			fmt.Printf("expected %d target labels, got %d\n", *flagNumOutput, len(labelSplits))
			os.Exit(1)
		}

		config := m.Config{
			Name:               networkName,
			InputNum:           *flagNumInputs,
			OutputNum:          *flagNumOutput,
			LayerNum:           *flagNumLayers,
			Epochs:             *flagNumEpochs,
			Activator:          activator,
			LearningRate:       *flagLearningRate,
			HiddenLayerNeurons: hiddenLayerNeurons,
			TargetLabels:       labelSplits,
			BatchSize:          *flagBatchSize,
			CipherIndex:        *flagCipherIndex,
			ENCRYPTED:          *flagENCRYPTED,
		}

		train(config, encryptionElements)

	case "predict":
		predictFlags := flag.NewFlagSet("predict", flag.ContinueOnError)
		flagQuery := predictFlags.String("query", "0,1,0,0", "labels are name to call each output")
		err := predictFlags.Parse(os.Args[3:])
		if err != nil {
			fmt.Printf("parsing train flags: %s\n", err.Error())
			os.Exit(1)
		}
		queryStrings := strings.Split(*flagQuery, ",")
		query := make([]float64, len(queryStrings))
		for i, s := range queryStrings {
			num, err := strconv.ParseFloat(s, 64)
			if err != nil {
				fmt.Printf("parsing input: %s\n", err.Error())
			}
			query[i] = num
		}

		network, err := m.BestNetworkFor(networkName)
		if err != nil {
			fmt.Printf("predicting %s: %s\n", queryStrings, err.Error())
			os.Exit(1)
		}

		prediction := network.Predict(query, encryptionElements)
		fmt.Println("Prediction:", prediction)
	}
}

func train(config m.Config, encryptionElements m.EncryptionElements) {

	filename := "./data/mnist_train.csv"

	network := m.NewNetwork(config)

	lines, err := m.GetLinesMNIST(filename, config.InputNum, config.OutputNum)
	fmt.Println("Read lines...")

	if err != nil {
		fmt.Printf("couldn't get lines from file: %s\n", err.Error())
		os.Exit(1)
	}

	err = network.Train(lines, 0.1, 60, encryptionElements, config.ENCRYPTED, config.CipherIndex-1) //data,learn rate, batch size

	if err != nil {
		fmt.Printf("training network: %s\n", err.Error())
		os.Exit(1)
	}
	err = network.Analyze(encryptionElements)
	if err != nil {
		fmt.Printf("doing analysis of network: %s\n", err.Error())
		os.Exit(1)
	}
	fmt.Println("Training complete")
}

func parseHiddenLayers(hiddenLayersStr string) ([]int, error) {
	hiddenLayers := strings.Split(hiddenLayersStr, ",")
	neurons := make([]int, len(hiddenLayers))
	for i, str := range hiddenLayers {
		neuron, err := strconv.Atoi(str)
		if err != nil {
			return nil, err
		}
		neurons[i] = neuron
	}
	return neurons, nil
}
