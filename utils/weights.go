package utils

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"

	"cure_lib/tensor"
)

// WeightData represents serializable weight data for a layer
type WeightData struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	Data  []float64 `json:"data"`
}

// ModelWeights represents all weights in a model
type ModelWeights struct {
	Version string                 `json:"version"`
	Layers  map[string]LayerWeight `json:"layers"`
}

// LayerWeight contains weights and bias for a layer
type LayerWeight struct {
	Weight *WeightData `json:"weight,omitempty"`
	Bias   *WeightData `json:"bias,omitempty"`
}

// SaveWeights saves model weights to a JSON file
func SaveWeights(filepath string, weights *ModelWeights) error {
	data, err := json.MarshalIndent(weights, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal weights: %w", err)
	}
	return os.WriteFile(filepath, data, 0644)
}

// LoadWeights loads model weights from a JSON file
func LoadWeights(filepath string) (*ModelWeights, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read weights file: %w", err)
	}
	var weights ModelWeights
	if err := json.Unmarshal(data, &weights); err != nil {
		return nil, fmt.Errorf("failed to unmarshal weights: %w", err)
	}
	return &weights, nil
}

// TensorToWeightData converts a tensor to serializable weight data
func TensorToWeightData(name string, t *tensor.Tensor) *WeightData {
	return &WeightData{
		Name:  name,
		Shape: t.Shape,
		Data:  append([]float64{}, t.Data...), // copy
	}
}

// WeightDataToTensor converts weight data back to a tensor
func WeightDataToTensor(wd *WeightData) *tensor.Tensor {
	t := tensor.New(wd.Shape...)
	copy(t.Data, wd.Data)
	return t
}

// CiphertextData represents serializable ciphertext (base64 encoded)
type CiphertextData struct {
	Level int    `json:"level"`
	Scale string `json:"scale"`
	Data  string `json:"data"` // base64 encoded
}

// EncodeBytes encodes raw bytes to base64 string
func EncodeBytes(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

// DecodeBytes decodes base64 string to raw bytes
func DecodeBytes(encoded string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(encoded)
}
