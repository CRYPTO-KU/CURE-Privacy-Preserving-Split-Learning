package utils

import (
	"fmt"
	"io"
	"os"
	"time"
)

// Verbose controls whether timing statistics are printed.
// Set to false to suppress output.
var Verbose = true

// Output is the writer where timing statistics are printed.
// Defaults to os.Stdout.
var Output io.Writer = os.Stdout

// TimingStats holds timing information for different operations
type TimingStats struct {
	TotalTime            time.Duration
	DataLoadingTime      time.Duration
	HEInitTime           time.Duration
	ModelInitTime        time.Duration
	ForwardPassTime      time.Duration
	BackwardPassTime     time.Duration
	UpdateTime           time.Duration
	EncryptionTime       time.Duration
	DecryptionTime       time.Duration
	ServerLinearTime     time.Duration
	ServerActivationTime time.Duration
	ClientLinearTime     time.Duration
	ClientActivationTime time.Duration
	LossComputationTime  time.Duration
}

// PrintTimingStats prints detailed timing statistics.
// Respects the Verbose flag - does nothing if Verbose is false.
func PrintTimingStats(stats *TimingStats, steps int) {
	if !Verbose {
		return
	}
	fmt.Fprintln(Output, "\n=== TIMING STATISTICS ===")
	fmt.Fprintf(Output, "Total training time: %v\n", stats.TotalTime)
	fmt.Fprintf(Output, "Average time per step: %v\n", stats.TotalTime/time.Duration(steps))
	fmt.Fprintf(Output, "Steps completed: %d\n", steps)
	fmt.Fprintln(Output, "\nBreakdown by operation:")
	fmt.Fprintf(Output, "  Data loading: %v (%.1f%%)\n", stats.DataLoadingTime, float64(stats.DataLoadingTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  HE initialization: %v (%.1f%%)\n", stats.HEInitTime, float64(stats.HEInitTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  Model initialization: %v (%.1f%%)\n", stats.ModelInitTime, float64(stats.ModelInitTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  Forward pass: %v (%.1f%%)\n", stats.ForwardPassTime, float64(stats.ForwardPassTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  Backward pass: %v (%.1f%%)\n", stats.BackwardPassTime, float64(stats.BackwardPassTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  Weight updates: %v (%.1f%%)\n", stats.UpdateTime, float64(stats.UpdateTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  Encryption: %v (%.1f%%)\n", stats.EncryptionTime, float64(stats.EncryptionTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  Decryption: %v (%.1f%%)\n", stats.DecryptionTime, float64(stats.DecryptionTime)/float64(stats.TotalTime)*100)
	fmt.Fprintf(Output, "  Loss computation: %v (%.1f%%)\n", stats.LossComputationTime, float64(stats.LossComputationTime)/float64(stats.TotalTime)*100)
	fmt.Fprintln(Output, "\nForward pass breakdown:")
	fmt.Fprintf(Output, "  Server Linear: %v (%.1f%% of forward)\n", stats.ServerLinearTime, float64(stats.ServerLinearTime)/float64(stats.ForwardPassTime)*100)
	fmt.Fprintf(Output, "  Server Activation: %v (%.1f%% of forward)\n", stats.ServerActivationTime, float64(stats.ServerActivationTime)/float64(stats.ForwardPassTime)*100)
	fmt.Fprintf(Output, "  Client Linear: %v (%.1f%% of forward)\n", stats.ClientLinearTime, float64(stats.ClientLinearTime)/float64(stats.ForwardPassTime)*100)
	fmt.Fprintf(Output, "  Client Activation: %v (%.1f%% of forward)\n", stats.ClientActivationTime, float64(stats.ClientActivationTime)/float64(stats.ForwardPassTime)*100)
	fmt.Fprintln(Output, "\nPerformance metrics:")
	fmt.Fprintf(Output, "  Average forward pass time: %v\n", stats.ForwardPassTime/time.Duration(steps))
	fmt.Fprintf(Output, "  Average backward pass time: %v\n", stats.BackwardPassTime/time.Duration(steps))
	fmt.Fprintf(Output, "  Average encryption time: %v\n", stats.EncryptionTime/time.Duration(steps))
	fmt.Fprintf(Output, "  Average decryption time: %v\n", stats.DecryptionTime/time.Duration(steps))
}

// DurationUS converts any time.Duration to micro-seconds as float64
func DurationUS(d time.Duration) float64 {
	return float64(d.Nanoseconds()) / 1_000.0
}
