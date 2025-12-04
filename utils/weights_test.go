package utils

import (
	"os"
	"path/filepath"
	"testing"

	"cure_lib/tensor"
)

func TestTensorToWeightData(t *testing.T) {
	// Create a test tensor
	ten := tensor.New(2, 3)
	for i := range ten.Data {
		ten.Data[i] = float64(i) * 0.5
	}

	// Convert to weight data
	wd := TensorToWeightData("test_weight", ten)

	// Verify
	if wd.Name != "test_weight" {
		t.Errorf("Name = %s, want test_weight", wd.Name)
	}
	if len(wd.Shape) != 2 || wd.Shape[0] != 2 || wd.Shape[1] != 3 {
		t.Errorf("Shape = %v, want [2, 3]", wd.Shape)
	}
	if len(wd.Data) != 6 {
		t.Errorf("Data length = %d, want 6", len(wd.Data))
	}
	for i, v := range wd.Data {
		expected := float64(i) * 0.5
		if v != expected {
			t.Errorf("Data[%d] = %f, want %f", i, v, expected)
		}
	}
}

func TestWeightDataToTensor(t *testing.T) {
	wd := &WeightData{
		Name:  "test",
		Shape: []int{3, 4},
		Data:  make([]float64, 12),
	}
	for i := range wd.Data {
		wd.Data[i] = float64(i)
	}

	ten := WeightDataToTensor(wd)

	if len(ten.Shape) != 2 || ten.Shape[0] != 3 || ten.Shape[1] != 4 {
		t.Errorf("Shape = %v, want [3, 4]", ten.Shape)
	}
	for i, v := range ten.Data {
		if v != float64(i) {
			t.Errorf("Data[%d] = %f, want %f", i, v, float64(i))
		}
	}
}

func TestSaveLoadWeights(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "weights_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	weightsFile := filepath.Join(tmpDir, "test_weights.json")

	// Create test weights
	weights := &ModelWeights{
		Version: "1.0",
		Layers: map[string]LayerWeight{
			"layer1": {
				Weight: &WeightData{
					Name:  "layer1_weight",
					Shape: []int{128, 784},
					Data:  make([]float64, 128*784),
				},
				Bias: &WeightData{
					Name:  "layer1_bias",
					Shape: []int{128},
					Data:  make([]float64, 128),
				},
			},
			"layer2": {
				Weight: &WeightData{
					Name:  "layer2_weight",
					Shape: []int{10, 128},
					Data:  make([]float64, 10*128),
				},
			},
		},
	}

	// Initialize with some values
	for i := range weights.Layers["layer1"].Weight.Data {
		weights.Layers["layer1"].Weight.Data[i] = float64(i) * 0.001
	}
	for i := range weights.Layers["layer1"].Bias.Data {
		weights.Layers["layer1"].Bias.Data[i] = float64(i) * 0.01
	}

	// Save
	err = SaveWeights(weightsFile, weights)
	if err != nil {
		t.Fatalf("SaveWeights failed: %v", err)
	}

	// Load
	loaded, err := LoadWeights(weightsFile)
	if err != nil {
		t.Fatalf("LoadWeights failed: %v", err)
	}

	// Verify
	if loaded.Version != "1.0" {
		t.Errorf("Version = %s, want 1.0", loaded.Version)
	}
	if len(loaded.Layers) != 2 {
		t.Errorf("Layers count = %d, want 2", len(loaded.Layers))
	}

	layer1 := loaded.Layers["layer1"]
	if layer1.Weight == nil {
		t.Fatal("layer1 weight is nil")
	}
	if len(layer1.Weight.Shape) != 2 || layer1.Weight.Shape[0] != 128 || layer1.Weight.Shape[1] != 784 {
		t.Errorf("layer1 weight shape = %v, want [128, 784]", layer1.Weight.Shape)
	}

	// Check data values
	if layer1.Weight.Data[0] != 0.0 {
		t.Errorf("layer1.Weight.Data[0] = %f, want 0", layer1.Weight.Data[0])
	}
	if layer1.Weight.Data[1] != 0.001 {
		t.Errorf("layer1.Weight.Data[1] = %f, want 0.001", layer1.Weight.Data[1])
	}
}

func TestEncodeDecodeBytes(t *testing.T) {
	original := []byte("test binary data with special chars: \x00\x01\x02")

	encoded := EncodeBytes(original)
	decoded, err := DecodeBytes(encoded)
	if err != nil {
		t.Fatalf("DecodeBytes failed: %v", err)
	}

	if string(decoded) != string(original) {
		t.Errorf("Round-trip failed: got %v, want %v", decoded, original)
	}
}

func TestLoadWeightsNotFound(t *testing.T) {
	_, err := LoadWeights("/nonexistent/path/weights.json")
	if err == nil {
		t.Error("Expected error for nonexistent file")
	}
}

func TestLoadWeightsInvalidJSON(t *testing.T) {
	// Create temp file with invalid JSON
	tmpDir, err := os.MkdirTemp("", "weights_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	badFile := filepath.Join(tmpDir, "bad.json")
	err = os.WriteFile(badFile, []byte("not valid json"), 0644)
	if err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	_, err = LoadWeights(badFile)
	if err == nil {
		t.Error("Expected error for invalid JSON")
	}
}
