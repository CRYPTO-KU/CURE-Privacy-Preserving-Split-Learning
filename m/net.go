package m

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"

	// "github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"gonum.org/v1/gonum/mat"
)

type Config struct {
	Name               string
	InputNum           int
	HiddenNum          int
	OutputNum          int
	LayerNum           int
	Epochs             int
	TargetLabels       []string
	Activator          Activator
	LearningRate       float64
	HiddenLayerNeurons []int // New field to hold the number of neurons in each hidden layer
	BatchSize          int
	CipherIndex        int
	ENCRYPTED          bool
}

type EncryptionElements struct {
	Params    hefloat.Parameters
	Encoder   *hefloat.Encoder
	Encryptor *rlwe.Encryptor
	Decryptor *rlwe.Decryptor
	Evaluator *hefloat.Evaluator
	// btpEvaluator *bootstrapping.Evaluator
}

type CryptoSystem struct {
	eval      *hefloat.Evaluator
	encoder   *hefloat.Encoder
	encryptor *rlwe.Encryptor
	decyrptor *rlwe.Decryptor
	params    hefloat.Parameters
}

func newPredictionNetwork(weights []mat.Matrix, run runInfo) Network {
	return Network{
		config: Config{
			Activator:    run.activator,
			TargetLabels: run.targetLabels,
		},
		weights:      weights,
		layers:       make([]mat.Matrix, len(weights)+1),
		weightedSums: make([]mat.Matrix, len(weights)),
	}
}

func NewNetwork(c Config) Network {
	totalWeights := len(c.HiddenLayerNeurons) + 1 // Number of hidden layers + output layer
	net := Network{
		config:       c,
		weights:      make([]mat.Matrix, totalWeights),
		layers:       make([]mat.Matrix, len(c.HiddenLayerNeurons)+2), // Input, hidden layers, and output
		weightedSums: make([]mat.Matrix, totalWeights),
		errors:       make([]mat.Matrix, len(c.HiddenLayerNeurons)+2),
	}

	lastWeightIndex := len(net.weights) - 1
	for i := 0; i <= lastWeightIndex; i++ {
		var rows, cols int
		if i == 0 { // Input layer to the first hidden layer
			rows = c.HiddenLayerNeurons[0]
			cols = c.InputNum
		} else if i == lastWeightIndex { // Last hidden layer to output layer
			rows = c.OutputNum
			cols = c.HiddenLayerNeurons[len(c.HiddenLayerNeurons)-1]
		} else { // Hidden layer to hidden layer
			rows = c.HiddenLayerNeurons[i]
			cols = c.HiddenLayerNeurons[i-1]
		}

		net.weights[i] = mat.NewDense(rows, cols, randomArray(rows*cols, float64(cols)))
	}

	return net
}

func NewServer(c Config) Network {
	totalWeights := 1
	net := Network{
		config:       c,
		weights:      make([]mat.Matrix, totalWeights),
		layers:       make([]mat.Matrix, 2), // Input, hidden layers, and output
		weightedSums: make([]mat.Matrix, totalWeights),
		errors:       make([]mat.Matrix, 2),
	}

	net.weights[0] = mat.NewDense(784, 128, randomArray(784*128, float64(128)))

	return net
}

type Network struct {
	trainingStart int64
	trainingEnd   int64
	weights       []mat.Matrix
	layers        []mat.Matrix
	weightedSums  []mat.Matrix
	errors        []mat.Matrix
	config        Config
}

func (net Network) lastIndex() int {
	return len(net.layers) - 1
}

func GetFirstLayer(net Network) []float64 {
	return GetColumn(net.weights[0], 0)
}

func GetLastLayer(net Network) []float64 {
	return GetColumn(net.weights[net.lastIndex()], 0)
}

func GetColumn(matrix mat.Matrix, j int) []float64 {
	rows, _ := matrix.Dims()
	firstColumn := make([]float64, rows)

	for i := 0; i < rows; i++ {
		firstColumn[i] = matrix.At(i, j)
	}

	return firstColumn
}

func (net Network) testFilepath() string {
	return path.Join("data", "test", net.config.Name+".data")
}

func (net Network) testExists() bool {
	info, err := os.Stat(net.testFilepath())
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func (net *Network) Train(lines Lines, learningRate float64, batchSize int, encryptionElements EncryptionElements, ENCRYPTED bool, cipher_index int) error {
	net.trainingStart = time.Now().Unix()
	fmt.Println("Started training...")

	batches := createBatches(lines, batchSize)
	for epoch := 1; epoch <= net.config.Epochs; epoch++ {
		batchNum := 1
		//bar := progressbar.Default(int64(len(batches)))
		for _, batch := range batches {
			net.trainBatchSGD(batch, learningRate, encryptionElements, ENCRYPTED, cipher_index)
			if batchNum == 60 {
			}
			batchNum++
			//bar.Add(1)
		}

		fmt.Printf("Epoch %d of %d complete\n", epoch, net.config.Epochs)
		err2 := net.Analyze(encryptionElements)
		if err2 != nil {
			fmt.Printf("doing analysis of network: %s\n", err2.Error())
			os.Exit(1)
		}
	}

	net.trainingEnd = time.Now().Unix()
	fmt.Printf("Training took %d seconds\n", net.trainingEnd-net.trainingStart)

	return nil
}

func (net *Network) trainBatchSGD(batch []Line, learningRate float64, encryptionElements EncryptionElements, ENCRYPTED bool, cipher_index int) {
	for _, line := range batch {
		net.trainOneSGD(line.Inputs, line.Targets, learningRate, encryptionElements, ENCRYPTED, cipher_index)
	}
}

func createBatches(lines Lines, batchSize int) [][]Line {
	numBatches := (len(lines) + batchSize - 1) / batchSize
	batches := make([][]Line, numBatches)

	for i := 0; i < numBatches; i++ {
		startIdx := i * batchSize
		endIdx := startIdx + batchSize

		if endIdx > len(lines) {
			endIdx = len(lines)
		}

		batch := lines[startIdx:endIdx]
		batches[i] = batch
	}

	return batches
}

func (net *Network) trainOneSGD(inputData []float64, targetData []float64, learningRate float64, encryptionElements EncryptionElements, ENCRYPTED bool, cipher_index int) {
	// Perform a single forward pass and backpropagation for the given data point
	net.feedForward(inputData, encryptionElements, ENCRYPTED, cipher_index)
	finalOutputs := net.layers[net.lastIndex()]
	net.backpropagate(targetData, finalOutputs, encryptionElements, ENCRYPTED, cipher_index)
}

func (net *Network) backpropagateOneLevelScalar(targetData []float64, finalOutputs mat.Matrix, encryptionElements EncryptionElements, ENCRYPTED bool, cipher_index int) {

	targets := mat.NewDense(len(targetData), 1, targetData)
	net.errors[len(net.errors)-1] = subtract(targets, finalOutputs)

	// Backpropagate the error through the hidden layers
	for i := net.lastIndex() - 1; i > 1; i-- {
		// Calculate the error for the current layer
		net.errors[i] = dot(net.weights[i].T(), net.errors[i+1])
	}
	//SERVER SIDE
	/*------------------------------------------------------------------------------------------------------------------*/
	net.errors[1] = dot(net.weights[1].T(), net.errors[2])
	/*------------------------------------------------------------------------------------------------------------------*/

	// Perform weight updates using SGD for each layer
	for i := net.lastIndex(); i > 1; i-- {
		// Compute the gradients for the current layer based on the error and activation function
		sigmoid := ActivatorLookup["sigmoid"]
		if i == net.lastIndex() {
			// last layer
			gradients := multiply(net.errors[i], sigmoid.Deactivate(net.layers[i]))
			weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[i-1].T()))
			net.weights[i-1] = add(net.weights[i-1], weightUpdate).(*mat.Dense)
		} else {
			// intermediate layers
			gradients := multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i]))
			weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[i-1].T()))
			net.weights[i-1] = add(net.weights[i-1], weightUpdate).(*mat.Dense)
		}
	}
	//SERVER SIDE
	/*------------------------------------------------------------------------------------------------------------------*/

	gradients := multiply(net.errors[1], net.config.Activator.Deactivate(net.layers[1]))
	weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[0].T()))

	if !ENCRYPTED {

		for i := net.lastIndex() - 1; i > 0; i-- {
			// Calculate the error for the current layer
			net.errors[i] = dot(net.weights[i].T(), net.errors[i+1])
		}
		//SERVER SIDE

		net.errors[1] = dot(net.weights[1].T(), net.errors[2])

		// Perform weight updates using SGD for each layer
		for i := net.lastIndex(); i > 1; i-- {
			// Compute the gradients for the current layer based on the error and activation function
			sigmoid := ActivatorLookup["sigmoid"]

			if i == net.lastIndex() {

				// last layer
				gradients := multiply(net.errors[i], sigmoid.Deactivate(net.layers[i]))
				weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[i-1].T()))
				net.weights[i-1] = add(net.weights[i-1], weightUpdate).(*mat.Dense)

			} else {

				// intermediate layers
				gradients := multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i]))
				weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[i-1].T()))
				net.weights[i-1] = add(net.weights[i-1], weightUpdate).(*mat.Dense)
			}
		}

		net.weights[0] = add(net.weights[0], weightUpdate).(*mat.Dense)

	} else {

		params := encryptionElements.Params
		encoder := encryptionElements.Encoder
		encryptor := encryptionElements.Encryptor
		decryptor := encryptionElements.Decryptor
		evaluator := encryptionElements.Evaluator

		numRowsWeights, _ := net.weights[0].Dims()
		_, numColsUpdate := weightUpdate.Dims()
		floatWeights := MatrixToVector(net.weights[0])
		floatUpdate := MatrixToVector(weightUpdate)

		splitAmount := 15
		floatWeightsSplit := splitFloatsIntoParts(floatWeights, splitAmount)
		floatUpdateSplit := splitFloatsIntoParts(floatUpdate, splitAmount)

		encodedWeightsSplit := make([]*rlwe.Plaintext, splitAmount)
		encodedUpdatesSplit := make([]*rlwe.Plaintext, splitAmount)
		for i := 0; i < splitAmount; i++ {
			encodedWeightsSplit[i] = hefloat.NewPlaintext(params, params.MaxLevel())
			encodedUpdatesSplit[i] = hefloat.NewPlaintext(params, params.MaxLevel())
			if err := encoder.Encode(floatWeightsSplit[i], encodedWeightsSplit[i]); err != nil {
				panic(err)
			}
			if err := encoder.Encode(floatUpdateSplit[i], encodedUpdatesSplit[i]); err != nil {
				panic(err)
			}
		}

		encryptedWeightsSplit := make([]*rlwe.Ciphertext, splitAmount)
		encryptedUpdatesSplit := make([]*rlwe.Ciphertext, splitAmount)
		for i := 0; i < splitAmount; i++ {
			var err error // Declare err inside the loop
			// Encrypt weights
			encryptedWeightsSplit[i], err = encryptor.EncryptNew(encodedWeightsSplit[i])
			if err != nil {
				panic(err)
			}

			// Encrypt updates
			encryptedUpdatesSplit[i], err = encryptor.EncryptNew(encodedUpdatesSplit[i])
			if err != nil {
				panic(err)
			}
		}

		for i := 0; i < splitAmount; i++ {
			evaluator.Add(encryptedUpdatesSplit[i], encryptedWeightsSplit[i], encryptedWeightsSplit[i])
		}

		decodedWeightsSplit := make([][]complex128, splitAmount, len(encodedWeightsSplit))
		// decodedWeightsSplit := make([][]complex128, splitAmount, len(encodedWeightsSplit[0].Value.Buff))
		for i := 0; i < splitAmount; i++ {
			encodedWeightsSplit[i] = decryptor.DecryptNew(encryptedWeightsSplit[i])
			valuesTest := make([]complex128, params.LogMaxSlots())
			if err := encoder.Decode(encodedWeightsSplit[i], valuesTest); err != nil {
				panic(err)
			}

			decodedWeightsSplit[i] = valuesTest
		}

		weightsSplit := make([][]float64, splitAmount)
		for i := range weightsSplit {
			weightsSplit[i] = make([]float64, len(decodedWeightsSplit[0]))
		}

		for i := 0; i < splitAmount; i++ {
			for j, val := range decodedWeightsSplit[i] {
				weightsSplit[i][j] = real(val)
			}
		}

		weights := make([]float64, len(weightsSplit[0])*len(weightsSplit))
		for i := 0; i < splitAmount; i++ {
			copy(weights[len(weightsSplit[0])*i:], weightsSplit[i])
		}

		// Calculate the new capacity required for the slice
		newCapacity := numRowsWeights * numColsUpdate

		// Allocate a new slice with sufficient capacity
		newWeights := make([]float64, newCapacity)

		// Copy the elements from the original slice to the new slice
		copy(newWeights, weights)

		// Perform the slicing operation
		newWeights = newWeights[:numRowsWeights*numColsUpdate]

		// Assign the sliced weights back to net.weights[0]
		net.weights[0] = VectorToMatrix(newWeights, numRowsWeights, numColsUpdate)

	}
	/*------------------------------------------------------------------------------------------------------------------*/
}

func (net *Network) backpropagate(targetData []float64, finalOutputs mat.Matrix, encryptionElements EncryptionElements, ENCRYPTED bool, cipher_index int) {
	for i := net.lastIndex(); i > cipher_index; i-- {
		// find errors
		if i == net.lastIndex() {
			// network error
			targets := mat.NewDense(len(targetData), 1, targetData)
			net.errors[len(net.errors)-1] = subtract(targets, finalOutputs)
		} else {
			net.errors[i] = dot(net.weights[i].T(), net.errors[i+1])
		}
		net.weights[i-1] = add(net.weights[i-1],
			scale(net.config.LearningRate,
				dot(multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i])),
					net.layers[i-1].T()))).(*mat.Dense)
	}

	if ENCRYPTED {
		for i := cipher_index; i > 0; i-- {
			// find errors
			if i == net.lastIndex() {
				// network error
				targets := mat.NewDense(len(targetData), 1, targetData)
				net.errors[len(net.errors)-1] = subtract(targets, finalOutputs)
			} else {
				errors := mat.DenseCopyOf(net.weights[i].T())
				layers := mat.DenseCopyOf(net.errors[i+1])
				net.errors[i] = CCmatrixmatrixDotParalellized(errors, layers, encryptionElements)
			}
			errors := mat.DenseCopyOf(multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i])))
			layers := mat.DenseCopyOf(net.layers[i-1].T())
			net.weights[i-1] = add(net.weights[i-1],
				scale(net.config.LearningRate,
					CCmatrixmatrixDotParalellized(errors,
						layers, encryptionElements))).(*mat.Dense)
		}
	} else {
		for i := cipher_index; i > 0; i-- {
			// find errors
			if i == net.lastIndex() {
				// network error
				targets := mat.NewDense(len(targetData), 1, targetData)
				net.errors[len(net.errors)-1] = subtract(targets, finalOutputs)
			} else {
				net.errors[i] = dot(net.weights[i].T(), net.errors[i+1])
			}
			net.weights[i-1] = add(net.weights[i-1],
				scale(net.config.LearningRate,
					dot(multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i])),
						net.layers[i-1].T()))).(*mat.Dense)
		}
	}
}

func (net *Network) feedForward(inputData []float64, encryptionElems EncryptionElements, ENCRYPTED bool, cipher_index int) {
	sigmoid := ActivatorLookup["sigmoid"]

	// --- begin SERVER ---
	net.layers[0] = mat.NewDense(len(inputData), 1, inputData)

	// --- UNENCRYPTED and WITH NOISE (for testing)
	if !ENCRYPTED {

		for i := 0; i < cipher_index; i++ {
			if i != len(net.layers)-1 {
				net.weightedSums[i] = dot(net.weights[i], net.layers[i])
			}

			// do sigmoid for the last layer
			if i == len(net.layers)-2 {
				net.layers[i+1] = apply(sigmoid.Activate, net.weightedSums[i])
				//net.layers[i], _ = ApplySoftmaxToColumns(net.layers[i]) // applying softmax in the end
			} else {
				net.layers[i+1] = apply(net.config.Activator.Activate, net.weightedSums[i])
			}
		}
	} else {
		/* ONE LEVEL SCALAR IMPLEMENTATION
			params := encryptionElems.Params
			encoder := encryptionElems.Encoder
			encryptor := encryptionElems.Encryptor
			decryptor := encryptionElems.Decryptor
			evaluator := encryptionElems.Evaluator
			denseMatrix1, err1 := net.weights[0].(*mat.Dense)
			if !err1 {
				// Handle the case where net.weights[0] is not of type *mat.Dense
				fmt.Println("net.weights[0] is not of type *mat.Dense")
				return // or handle the error as appropriate
			}

			denseMatrix2, err2 := net.layers[0].(*mat.Dense)
			if !err2 {
				// Handle the case where net.weights[0] is not of type *mat.Dense
				fmt.Println("net.weights[0] is not of type *mat.Dense")
				return
			}

			values := make([]float64, 128)
			values = OneLevelScalarMultiThread(encoder, encryptor, *evaluator, decryptor, params, denseMatrix1, denseMatrix2)
			net.weightedSums[0] = mat.NewDense(128, 1, values)
			net.layers[1] = apply(net.config.Activator.Activate, net.weightedSums[0])
			// print("net.layers", len(net.layers))
		}*/

		/*CC matrix Implementation*/
		for i := 0; i < cipher_index; i++ {
			if i != len(net.layers)-1 {
				denseWeights := mat.DenseCopyOf(net.weights[i])
				denseLayers := mat.DenseCopyOf(net.layers[i])
				net.weightedSums[i] = CCmatrixmatrixDotParalellized(denseWeights, denseLayers, encryptionElems)
			}

			// do sigmoid for the last layer
			if i == len(net.layers)-2 {
				net.layers[i+1] = apply(sigmoid.Activate, net.weightedSums[i])
				//net.layers[i], _ = ApplySoftmaxToColumns(net.layers[i]) // applying softmax in the end
			} else {
				net.layers[i+1] = apply(net.config.Activator.Activate, net.weightedSums[i])
			}
		}
	}
	for i := cipher_index; i < len(net.layers)-1; i++ {
		if i != len(net.layers)-1 {
			net.weightedSums[i] = dot(net.weights[i], net.layers[i])
		}

		// do sigmoid for the last layer
		if i == len(net.layers)-2 {
			net.layers[i+1] = apply(sigmoid.Activate, net.weightedSums[i])
			//net.layers[i], _ = ApplySoftmaxToColumns(net.layers[i]) // applying softmax in the end
		} else {
			net.layers[i+1] = apply(net.config.Activator.Activate, net.weightedSums[i])
		}
	}

}

func (net Network) Predict(inputData []float64, encryptionElements EncryptionElements) string {
	// feedforward
	net.feedForward(inputData, encryptionElements, false, 2)

	bestOutputIndex := 0
	highest := 0.0
	outputs := net.layers[net.lastIndex()]
	for i := 0; i < net.config.OutputNum; i++ {
		if outputs.At(i, 0) > highest {
			bestOutputIndex = i
			highest = outputs.At(i, 0)
		}
	}
	return net.labelFor(bestOutputIndex)
}

var outPath = path.Join("data", "out")
var analysisFilepath = path.Join(outPath, "analysis.csv")

// Analyze tests the network against the test set and outputs the accuracy as well as writing to a log
func (net Network) Analyze(encryptionElements EncryptionElements) error {
	var needsHeaders bool
	err := os.MkdirAll(outPath, os.ModePerm)
	if _, err := os.Stat(analysisFilepath); os.IsNotExist(err) {
		needsHeaders = true
	}
	file, err := os.OpenFile(analysisFilepath,
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)

	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	if needsHeaders {
		err = w.Write([]string{
			"Name", "Activator", "Inputs", "Hiddens", "Outputs", "Layers", "Epochs", "Target Labels", "LR", "End Time", "SecondsToTrain", "Accuracy",
		})
		if err != nil {
			return fmt.Errorf("writing csv headers: %w", err)
		}
		w.Flush()
	}
	record := make([]string, 12)
	record[0] = net.config.Name
	record[1] = net.config.Activator.String()
	record[2] = strconv.Itoa(net.config.InputNum)
	record[3] = strconv.Itoa(net.config.HiddenNum)
	record[4] = strconv.Itoa(net.config.OutputNum)
	record[5] = strconv.Itoa(net.config.LayerNum)
	record[6] = strconv.Itoa(net.config.Epochs)
	record[7] = strings.Join(net.config.TargetLabels, ", ")
	record[8] = strconv.FormatFloat(net.config.LearningRate, 'f', 4, 32)
	record[9] = strconv.Itoa(int(net.trainingEnd))
	record[10] = strconv.Itoa(int(net.trainingEnd - net.trainingStart))

	accuracy, err := net.test(encryptionElements)
	if err != nil {
		return fmt.Errorf("testing network: %w", err)
	}
	record[11] = strconv.FormatFloat(accuracy, 'f', 5, 32)
	fmt.Printf("Accuracy %.2f%%\n", accuracy)
	err = w.Write(record)
	if err := w.Error(); err != nil {
		return fmt.Errorf("error writing csv: %s", err.Error())
	}
	w.Flush()

	return nil
}

func (net Network) test(encryptionElements EncryptionElements) (float64, error) {
	var correct float64
	var total float64
	filename := "./pytorch_experiments/mnist_test.csv"

	lines, err := GetLinesMNIST(filename, net.config.InputNum, net.config.OutputNum)

	if err != nil {
		return 0, fmt.Errorf("getting lines: %w", err)
	}
	total = float64(len(lines))
	for _, line := range lines {
		prediction := net.Predict(line.Inputs, encryptionElements)
		var actual string
		for i, t := range line.Targets {
			if int(t+math.Copysign(0.5, t)) == 1 {
				actual = net.labelFor(i)
				break
			}
		}
		if actual == prediction {
			correct++
		}
	}

	percent := 100 * (correct / total)

	return percent, nil
}

func (net Network) labelFor(index int) string {
	return net.config.TargetLabels[index]
}

func (net Network) save() error {
	fmt.Printf("saving layer weight files for %s, run #%d\n", net.config.Name, net.trainingEnd)
	for i := 0; i < len(net.weights); i++ {
		filename := fmt.Sprintf("%s-%d-%d.wgt", net.config.Name, net.trainingEnd, i)
		f, err := os.Create(path.Join("data", "out", filename))
		if err != nil {
			return err
		}
		d := net.weights[i].(*mat.Dense)
		_, err = d.MarshalBinaryTo(f)
		if err != nil {
			return fmt.Errorf("marshalling weights: %w\n", err)
		}
		err = f.Close()
		if err != nil {
			return err
		}
	}

	return nil
}

func load(run runInfo) (Network, error) {
	sep := string(os.PathSeparator)
	pattern := fmt.Sprintf(".%s%s%s%s-%s-*.wgt", sep, outPath, sep, run.name, run.bestEndingTime)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return Network{}, fmt.Errorf("matching pattern %s: %w", pattern, err)
	}
	weights := make([]mat.Dense, len(matches))
	for _, m := range matches {
		splits := strings.Split(m, "-")
		layerString := strings.Split(splits[2], ".")[0]
		layerIndex, err := strconv.Atoi(layerString)
		if err != nil {
			return Network{}, fmt.Errorf("converting layer portion of filename to a number: %w", err)
		}
		f, err := os.Open(m)
		if err != nil {
			return Network{}, fmt.Errorf("opening file for layer %s: %w", layerString, err)
		}
		weights[layerIndex].Reset()
		_, err = weights[layerIndex].UnmarshalBinaryFrom(f)
		if err != nil {
			return Network{}, fmt.Errorf("unmarshalling layer %s: %w", layerString, err)
		}
		err = f.Close()
		if err != nil {
			return Network{}, fmt.Errorf("closing file for layer %s: %w", layerString, err)
		}
	}
	matrices := make([]mat.Matrix, len(weights))
	for i := range weights {
		matrices[i] = &weights[i]
	}

	return newPredictionNetwork(matrices, run), nil
}

type runInfo struct {
	name           string
	bestEndingTime string
	targetLabels   []string
	activator      Activator
}

const csvRecords = 12

// bestRun takes a dataset name and returns the best run epoch and activator
func bestRun(name string) (runInfo, error) {
	file, err := os.Open(analysisFilepath)
	if err != nil {
		return runInfo{}, fmt.Errorf("opening analysis csv file: %w", err)
	}
	r := csv.NewReader(file)
	// set to negative because if all accuracies for this data set were not measured, they failed parse of accuracy will
	// parse as the zero value of a float (0), which allows us to use the untested run until we get test data
	highestAccuracy := -1.
	var run runInfo
	i := 0
	// Iterate through the records
	for {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return runInfo{}, fmt.Errorf("reading record: %w", err)
		}
		if len(record) != csvRecords {
			if i == 0 {
				return runInfo{}, fmt.Errorf("there are %d analysis csv headers, expected %d", len(record), csvRecords)
			} else {
				return runInfo{}, fmt.Errorf("there are %d analysis csv values in record %d, expected %d", len(record), i, csvRecords)
			}
		}
		// record[0] is name
		// record[1] is activator
		// record[7] is the comma separated list of target labels
		// record[9] is time ending (epoch time)
		// record[11] is Accuracy
		if record[0] != name {
			continue
		}
		accuracy, _ := strconv.ParseFloat(record[11], 64)
		if accuracy > highestAccuracy {
			run.name = name
			highestAccuracy = accuracy
			run.bestEndingTime = record[9]
			var ok bool
			run.activator, ok = ActivatorLookup[record[1]]
			if !ok {
				return runInfo{}, fmt.Errorf("invalid activator: %s", record[1])
			}
			run.targetLabels = strings.Split(record[7], ",")
		}
		i++
	}

	return run, nil
}

func BestNetworkFor(name string) (Network, error) {
	run, err := bestRun(name)
	if err != nil {
		return Network{}, fmt.Errorf("getting best epoch for %s: %w", name, err)
	}
	net, err := load(run)
	if err != nil {
		return Network{}, fmt.Errorf("loading network")
	}

	return net, nil
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

func splitFloatsIntoParts(floats []float64, numParts int) [][]float64 {
	totalLength := len(floats)
	partSize := totalLength / numParts

	parts := make([][]float64, numParts)

	for i := 0; i < numParts; i++ {
		start := i * partSize
		end := (i + 1) * partSize

		if i == numParts-1 {
			// For the last part, include any remaining elements.
			end = totalLength
		}

		parts[i] = floats[start:end]
	}

	return parts
}

func (n *Network) SetWeights(newWeights []mat.Matrix) {
	if len(n.weights) != len(newWeights) {
		fmt.Println("Error: Mismatched number of weight matrices")
		return
	}

	for i := range n.weights {
		n.weights[i] = newWeights[i]
	}
}

func (n *Network) GetLayers() []mat.Matrix {
	return n.weights
}

func CCmatrixmatrixDotParalellized(mat1, mat2 *mat.Dense, cryptoSystem EncryptionElements) *mat.Dense {
	rows1, cols1 := mat1.Dims()
	rows2, cols2 := mat2.Dims()

	P := 1024 //number of slots of an rlwe interface of the encryption scheme

	mat1 = MatrixPadder(mat1)
	mat2 = MatrixPadder(mat2)

	padded_rows1, padded_cols1 := mat1.Dims()
	padded_rows2, padded_cols2 := mat2.Dims()

	if cols1 != rows2 {
		var err error
		panic(err)
	}

	batched_mat1 := reshapeMatrix(mat1, padded_rows1*padded_cols1/P, P)
	batched_mat2 := expandMatrix(mat2, P/padded_rows2)

	batched_rows1, _ := batched_mat1.Dims()
	//_, batched_cols2 := batched_mat2.Dims()

	var wg sync.WaitGroup
	wg.Add(batched_rows1)

	var data []float64

	for i := 0; i < batched_rows1; i++ {
		go func(i int) {
			defer wg.Done()
			for j := 0; j < cols2; j++ {
				var sum []float64
				vect1 := GetRowAsFloat(batched_mat1, i)
				vect2 := GetColumnAsFloat(batched_mat2, j)

				pt1 := hefloat.NewPlaintext(cryptoSystem.Params, cryptoSystem.Params.MaxLevel())
				if err := cryptoSystem.Encoder.Encode(vect1, pt1); err != nil {
					panic(err)
				}

				pt2 := hefloat.NewPlaintext(cryptoSystem.Params, cryptoSystem.Params.MaxLevel())
				if err := cryptoSystem.Encoder.Encode(vect2, pt2); err != nil {
					panic(err)
				}

				ct1, _ := cryptoSystem.Encryptor.EncryptNew(pt1)
				ct2, _ := cryptoSystem.Encryptor.EncryptNew(pt2)

				sum, _ = inner_product(ct1, ct2, cols1, cryptoSystem, P)
				data = append(data, sum...)
			}
		}(i)
	}

	wg.Wait()
	result := mat.NewDense(padded_rows1, padded_cols2, data)
	result = Submatrix(result, rows1, cols2)

	return result
}

func GetRowAsFloat(matrix *mat.Dense, rowIndex int) []float64 {
	_, cols := matrix.Dims()
	row := make([]float64, cols)
	for j := 0; j < cols; j++ {
		row[j] = matrix.At(rowIndex, j)
	}
	return row
}

func GetColumnAsFloat(matrix *mat.Dense, colIndex int) []float64 {
	rows, _ := matrix.Dims()
	column := make([]float64, rows)
	for i := 0; i < rows; i++ {
		column[i] = matrix.At(i, colIndex)
	}
	return column
}

func Submatrix(matrix *mat.Dense, rows, cols int) *mat.Dense {
	// Get the dimensions of the original matrix
	origRows, origCols := matrix.Dims()

	// Check if the requested rows and columns are within the bounds of the original matrix
	if rows > origRows || cols > origCols {
		panic("Requested submatrix dimensions exceed the dimensions of the original matrix")
	}

	// Extract the submatrix
	submatrix := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			submatrix.Set(i, j, matrix.At(i, j))
		}
	}

	return submatrix
}

func inner_product(ct1, ct2 *rlwe.Ciphertext, length_of_ct int, CryptoSystem EncryptionElements, P int) (result []float64, err error) {

	ct, _ := CryptoSystem.Evaluator.MulNew(ct1, ct2)
	CryptoSystem.Evaluator.Relinearize(ct, ct)
	var vect1 []complex128
	pt1 := hefloat.NewPlaintext(CryptoSystem.Params, CryptoSystem.Params.MaxLevel())
	if err := CryptoSystem.Encoder.Encode(vect1, pt1); err != nil {
		panic(err)
	}

	CryptoSystem.Encoder.Encode(vect1, pt1)
	ct_result, err := CryptoSystem.Encryptor.EncryptNew(pt1)

	CryptoSystem.Evaluator.Add(ct, ct_result, ct_result)

	for i := 1; i < length_of_ct; i = i * 2 {
		adder, _ := CryptoSystem.Evaluator.RotateNew(ct_result, i)
		CryptoSystem.Evaluator.Add(ct_result, adder, ct_result)
	}
	result_product := mask_first(ct_result, CryptoSystem, P)
	return result_product, nil
}

func reshapeMatrix(src *mat.Dense, newRows, newCols int) *mat.Dense {
	r, c := src.Dims()

	// Check if the total elements match
	if r*c != newRows*newCols {
		panic("new dimensions do not match the number of elements in the original matrix")
	}

	// Create a new matrix with the desired shape
	data := src.RawMatrix().Data
	dst := mat.NewDense(newRows, newCols, data)

	return dst
}

func expandMatrix(src *mat.Dense, repeats int) *mat.Dense {
	if repeats < 1 {
		panic("repeats must be at least 1")
	}

	r, c := src.Dims()
	newRows := r * repeats
	newData := make([]float64, newRows*c)

	// Fill new data by repeating the original matrix rows
	for i := 0; i < repeats; i++ {
		for j := 0; j < r; j++ {
			copy(newData[(i*r+j)*c:(i*r+j+1)*c], src.RawRowView(j))
		}
	}

	// Create a new Dense matrix with the repeated rows
	dst := mat.NewDense(newRows, c, newData)

	return dst
}

func mask_first(ct *rlwe.Ciphertext, cryptoSystem EncryptionElements, P int) []float64 {
	vect1 := []complex128{0 + 0i}
	pt := cryptoSystem.Decryptor.DecryptNew(ct)
	cryptoSystem.Encoder.Decode(pt, vect1)
	vectResult := maskValues(vect1, P)
	return vectResult
}

func maskValues(data []complex128, numSlots int) []float64 {
	var result []float64
	for i := 0; i < len(data); i += numSlots {
		result = append(result, real(data[i]))
	}
	return result
}

func MatrixPadder(matrix *mat.Dense) *mat.Dense {
	rows, cols := matrix.Dims()

	// Find the nearest power of 2 for the number of rows and columns
	paddedRows := int(math.Pow(2, math.Ceil(math.Log2(float64(rows)))))
	paddedCols := int(math.Pow(2, math.Ceil(math.Log2(float64(cols)))))

	// Create a new matrix with the padded dimensions
	paddedMatrix := mat.NewDense(paddedRows, paddedCols, nil)

	// Copy the original matrix into the padded matrix
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			paddedMatrix.Set(i, j, matrix.At(i, j))
		}
	}

	return paddedMatrix
}
