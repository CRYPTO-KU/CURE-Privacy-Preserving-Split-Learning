// cure-server: Server-side component for split learning with HE
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/split"
	"cure_lib/utils"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

var (
	logN      = flag.Int("logN", 13, "Ring dimension log2")
	modelType = flag.String("model", "mnist", "Model type: mnist, bcw")
	verbose   = flag.Bool("verbose", false, "Verbose output")
)

func main() {
	flag.Parse()
	utils.Verbose = *verbose

	log("CURE Server starting (logN=%d, model=%s)", *logN, *modelType)

	// Initialize HE
	heCtx := ckkswrapper.NewHeContextWithLogN(*logN)
	log("HE context ready")

	// Build server layers
	var serverLinear *layers.Linear
	var serverAct *layers.Activation

	switch *modelType {
	case "mnist":
		serverLinear = layers.NewLinear(784, 128, true, heCtx)
		serverAct, _ = layers.NewActivation("ReLU3", true, heCtx)
	case "bcw":
		serverLinear = layers.NewLinear(64, 32, true, heCtx)
		serverAct, _ = layers.NewActivation("ReLU3", true, heCtx)
	default:
		fmt.Fprintf(os.Stderr, "Unknown model: %s\n", *modelType)
		os.Exit(1)
	}

	// Init weights
	for i := range serverLinear.W.Data {
		serverLinear.W.Data[i] = 0.01 * float64(i%10-5)
	}
	serverLinear.SyncHE()
	log("Model ready")

	// Protocol
	protocol := split.NewProtocol(os.Stdin, os.Stdout)
	log("Waiting for client...")

	for {
		payload, err := protocol.ReceiveForward()
		if err == io.EOF {
			break
		}
		if err != nil {
			log("Error: %v", err)
			protocol.SendError(err)
			continue
		}

		log("Batch %d received", payload.BatchID)

		// Deserialize
		ct := new(rlwe.Ciphertext)
		if err := ct.UnmarshalBinary(payload.Ciphertext); err != nil {
			protocol.SendError(err)
			continue
		}

		// Forward
		ctOut, err := serverLinear.ForwardCipherMasked(ct)
		if err != nil {
			protocol.SendError(err)
			continue
		}
		ctOut, err = serverAct.ForwardCipher(ctOut)
		if err != nil {
			protocol.SendError(err)
			continue
		}

		// Bootstrap if needed
		if ckkswrapper.NeedsBootstrap(ctOut, 2) {
			ctOut, _ = heCtx.CheatBootstrap(ctOut)
		}

		// Send back
		ctBytes, _ := ctOut.MarshalBinary()
		protocol.SendForward(payload.BatchID, ctBytes, ctOut.Level(), 0)
		log("Batch %d sent", payload.BatchID)
	}

	log("Server done")
}

func log(format string, args ...interface{}) {
	if *verbose {
		fmt.Fprintf(os.Stderr, "[SERVER] "+format+"\n", args...)
	}
}
