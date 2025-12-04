package utils

import (
	"fmt"
	"strconv"
	"strings"
)

// Config holds training configuration
type Config struct {
	Architecture []int
	DataRoot     string
	BatchSize    int
	Steps        int
	HEMode       string
}

// ParseArchitecture parses architecture string into slice of integers
func ParseArchitecture(archStr string) ([]int, error) {
	archParts := strings.Fields(archStr)
	arch := make([]int, len(archParts))
	for i, s := range archParts {
		n, err := strconv.Atoi(s)
		if err != nil {
			return nil, err
		}
		arch[i] = n
	}
	return arch, nil
}

// ValidateConfig validates training configuration
func ValidateConfig(config *Config) error {
	if len(config.Architecture) < 2 {
		return fmt.Errorf("architecture must have at least 2 layers (input and output)")
	}

	if config.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive")
	}

	if config.Steps <= 0 {
		return fmt.Errorf("steps must be positive")
	}

	if config.HEMode != "split" {
		return fmt.Errorf("HE mode must be 'split'")
	}

	return nil
}
