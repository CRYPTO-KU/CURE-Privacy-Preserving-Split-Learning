package m

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type Line struct {
	Inputs  []float64
	Targets []float64
}
type Lines []Line

// first val in line is the label, rest are the pixel densities
func GetLinesMNIST(filename string, inputNum, outputNum int) (Lines, error) {
	var lines Lines
	file, _ := os.Open(filename)
	r := csv.NewReader(bufio.NewReader(file))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}

		inputs := make([]float64, inputNum)
		for i := range inputs {
			x, _ := strconv.ParseFloat(record[i+1], 64)
			inputs[i] = ((x / 255.0 * 0.99) + 0.01)
		}

		targets := make([]float64, 10)
		for i := range targets {
			targets[i] = 0.01
		}
		x, _ := strconv.Atoi(record[0])
		targets[x] = 0.99

		// TODO separate **train** inputs&targets into batches of fixed size (e.g. 64)
		// TODO shuffle  batches

		line := Line{
			Inputs:  inputs,
			Targets: targets,
		}
		lines = append(lines, line)
	}
	file.Close()

	return lines, nil
}

/*------------------------------------------------------------------------------------------------------------------------*/
func NormalizeLines(lines Lines, std []float64, mean []float64) Lines {
	normalizedLines := make(Lines, len(lines))
	for i, line := range lines {
		normalizedInputs := make([]float64, len(line.Inputs))
		for j, x := range line.Inputs {
			normalizedInputs[j] = (x - mean[j]) / std[j]
		}

		normalizedLines[i] = Line{
			Inputs:  normalizedInputs,
			Targets: line.Targets,
		}
	}
	return normalizedLines
}

func CalculateMean(lines Lines) []float64 {
	if len(lines) == 0 {
		return nil
	}

	numEntries := len(lines[0].Inputs)
	mean := make([]float64, numEntries)
	for _, line := range lines {
		for i, x := range line.Inputs {
			mean[i] += x
		}
	}

	for i := range mean {
		mean[i] /= float64(len(lines))
	}

	return mean
}

func CalculateStdDev(lines Lines) []float64 {
	if len(lines) == 0 {
		return nil
	}

	numEntries := len(lines[0].Inputs)

	mean := CalculateMean(lines)

	stdDev := make([]float64, numEntries)
	for _, line := range lines {
		for i, x := range line.Inputs {
			diff := x - mean[i]
			stdDev[i] += diff * diff
		}
	}

	for i := range stdDev {
		stdDev[i] = math.Sqrt(stdDev[i] / float64(len(lines)))
	}

	return stdDev
}

func GetLines(reader io.Reader, inputNum, outputNum int) (Lines, error) {
	scanner := bufio.NewScanner(reader)
	var lines Lines
	var lineNum int
	for scanner.Scan() {
		lineNum++
		splits := strings.Split(scanner.Text(), ",")
		if len(splits) != inputNum+1 {
			return lines, errInvalidLine{
				lineNum:  lineNum,
				splits:   len(splits),
				expected: inputNum + 1,
			}
		}
		inputs := make([]float64, inputNum)
		targets := make([]float64, outputNum)

		// goes over characters in one line of input
		for i, split := range splits {
			if i < inputNum {
				num, err := strconv.ParseFloat(split, 64)
				if err != nil {
					return lines, fmt.Errorf("parsing input: %w", err)
				}
				inputs[i] = num
			} else {
				num, err := strconv.ParseFloat(split, 64)
				if err != nil {
					return lines, fmt.Errorf("parsing target: %w", err)
				}
				targets[i-inputNum] = num
			}
		}
		line := Line{
			Inputs:  inputs,
			Targets: targets,
		}
		lines = append(lines, line)
	}
	return lines, nil
}

type errInvalidLine struct {
	lineNum  int
	splits   int
	expected int
}

func (e errInvalidLine) Error() string {
	return fmt.Sprintf("at line %d, expected %d values, got %d",
		e.lineNum, e.expected, e.splits)
}

func LineSplitter(batchSize, iterationNum int, lines Lines) Lines {
	start := batchSize * iterationNum
	end := batchSize * (iterationNum + 1)

	if start < 0 || start >= len(lines) || end <= start {
		// Return an empty slice if the indices are out of range
		return Lines{}
	}

	if end > len(lines) {
		// Adjust the end index if it exceeds the length of the lines
		end = len(lines)
	}

	return lines[start:end]
}
