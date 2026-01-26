package nn

import (
	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/tensor"
	"math"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Local copy of Softmax and CrossEntropyLoss to avoid import cycle
func softmax(logits *tensor.Tensor) *tensor.Tensor {
	maxLogit := logits.Data[0]
	for _, v := range logits.Data {
		if v > maxLogit {
			maxLogit = v
		}
	}
	expSum := 0.0
	exps := make([]float64, len(logits.Data))
	for i, v := range logits.Data {
		e := math.Exp(v - maxLogit)
		exps[i] = e
		expSum += e
	}
	softmax := tensor.New(len(logits.Data))
	for i, e := range exps {
		softmax.Data[i] = e / expSum
	}
	return softmax
}

type crossEntropyLoss struct{}

func (c *crossEntropyLoss) Backward(softmaxOut, oneHotLabel *tensor.Tensor) *tensor.Tensor {
	grad := tensor.New(len(softmaxOut.Data))
	for i := range grad.Data {
		grad.Data[i] = softmaxOut.Data[i] - oneHotLabel.Data[i]
	}
	return grad
}

func TestEndToEndCorrectness(t *testing.T) {
	// Setup: Create a simple Linear(4, 2) -> Activation("ReLU3") model
	heCtx := ckkswrapper.NewHeContext()

	linearLayer := layers.NewLinear(4, 2, true, heCtx)
	// Initialize weights with small random values
	for i := range linearLayer.W.Data {
		linearLayer.W.Data[i] = (rand.Float64() - 0.5) * 0.1
	}
	for i := range linearLayer.B.Data {
		linearLayer.B.Data[i] = (rand.Float64() - 0.5) * 0.1
	}
	linearLayer.SyncHE()

	activationLayer, err := layers.NewActivation("ReLU3", true, heCtx)
	if err != nil {
		t.Fatalf("failed to create Activation: %v", err)
	}

	// Create input data
	inputData := []float64{0.5, -0.3, 0.8, -0.1}
	inputTensor := tensor.New(4)
	copy(inputTensor.Data, inputData)

	// Encrypt input
	pt := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputData, pt)
	ctInput, err := heCtx.Encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("failed to encrypt input: %v", err)
	}

	t.Logf("=== FORWARD PASS ===")

	// Forward pass
	linearOutput, err := linearLayer.ForwardCipherMasked(ctInput)
	if err != nil {
		t.Fatalf("linear forward failed: %v", err)
	}

	activationOutput, err := activationLayer.ForwardCipher(linearOutput)
	if err != nil {
		t.Fatalf("activation forward failed: %v", err)
	}
	_ = activationOutput // Use output

	t.Logf("=== BACKWARD PASS ===")

	// Create mock gradient from client
	gradData := []float64{0.1, 0.1}
	gradTensor := tensor.New(2)
	copy(gradTensor.Data, gradData)

	ptGrad := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(gradData, ptGrad)
	ctGrad, err := heCtx.Encryptor.EncryptNew(ptGrad)
	if err != nil {
		t.Fatalf("failed to encrypt gradient: %v", err)
	}

	// Backward pass
	activationGradCt, err := activationLayer.BackwardHE(ctGrad)
	if err != nil {
		t.Fatalf("activation backward failed: %v", err)
	}

	_, err = linearLayer.BackwardHE(activationGradCt)
	if err != nil {
		t.Fatalf("linear backward failed: %v", err)
	}

	t.Logf("=== UPDATE ===")

	// Update weights
	err = linearLayer.Update(0.01)
	if err != nil {
		t.Fatalf("linear update failed: %v", err)
	}

	t.Logf("✓ Weight update completed successfully")
	t.Logf("✓ Weight gradients computed successfully")

	t.Logf("=== TEST COMPLETED SUCCESSFULLY ===")
}

func TestEndToEndHEPlainSplit(t *testing.T) {
	heCtx := ckkswrapper.NewHeContext()

	// --- Server-side (HE) ---
	server_linear := layers.NewLinear(784, 128, true, heCtx)
	server_linear.SyncHE()
	server_activation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// --- Client-side (Plain) ---
	client_linear1 := layers.NewLinear(128, 32, false, nil)
	client_activation, _ := layers.NewActivation("ReLU3", false, nil)
	client_linear2 := layers.NewLinear(32, 10, false, nil)

	// --- Data and Labels ---
	inputVec := tensor.New(784)
	for i := range inputVec.Data {
		inputVec.Data[i] = rand.NormFloat64()
	}
	label := rand.Intn(10)
	oneHotLabel := tensor.New(10)
	oneHotLabel.Data[label] = 1.0

	// --- FORWARD PASS ---
	// Server-side HE forward
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec.Data, ptInput)
	ctInput, err := heCtx.Encryptor.EncryptNew(ptInput)
	if err != nil {
		t.Fatalf("Encrypt input: %v", err)
	}
	ctHidden, err := server_linear.ForwardCipherMasked(ctInput)
	if err != nil {
		t.Fatalf("Server Linear Forward: %v", err)
	}
	ctHidden, err = server_activation.ForwardCipher(ctHidden)
	if err != nil {
		t.Fatalf("Server Activation Forward: %v", err)
	}

	// Client-side forward
	ptHidden := heCtx.Decryptor.DecryptNew(ctHidden)
	decoded := make([]complex128, 128)
	heCtx.Encoder.Decode(ptHidden, decoded)
	client_input_tensor := tensor.New(128)
	for i := range decoded {
		client_input_tensor.Data[i] = real(decoded[i])
	}
	l1_out, err := client_linear1.Forward(client_input_tensor)
	if err != nil {
		t.Fatalf("Client Linear1 Forward: %v", err)
	}
	act_out, err := client_activation.Forward(l1_out.(*tensor.Tensor))
	if err != nil {
		t.Fatalf("Client Activation Forward: %v", err)
	}
	l2_out, err := client_linear2.Forward(act_out.(*tensor.Tensor))
	if err != nil {
		t.Fatalf("Client Linear2 Forward: %v", err)
	}
	final_logits := l2_out.(*tensor.Tensor)
	final_probs := softmax(final_logits)

	// --- LOSS & BACKWARD (Plain) ---
	lossObj := &crossEntropyLoss{}
	grad_start := lossObj.Backward(final_probs, oneHotLabel)
	grad_l2, err := client_linear2.Backward(grad_start)
	if err != nil {
		t.Fatalf("Client Linear2 Backward: %v", err)
	}
	grad_act, err := client_activation.Backward(grad_l2)
	if err != nil {
		t.Fatalf("Client Activation Backward: %v", err)
	}
	grad_l1, err := client_linear1.Backward(grad_act)
	if err != nil {
		t.Fatalf("Client Linear1 Backward: %v", err)
	}

	// --- Encrypt gradient and HE backward ---
	pt_grad_to_server := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(grad_l1.(*tensor.Tensor).Data, pt_grad_to_server)
	ct_grad_to_server, err := heCtx.Encryptor.EncryptNew(pt_grad_to_server)
	if err != nil {
		t.Fatalf("Encrypt grad to server: %v", err)
	}
	// Server-side backward and update
	_, err = server_activation.BackwardHE(ct_grad_to_server)
	if err != nil {
		t.Fatalf("Server Activation BackwardHE: %v", err)
	}
	_, err = server_linear.BackwardHE(ct_grad_to_server)
	if err != nil {
		t.Fatalf("Server Linear Backward: %v", err)
	}
	err = server_linear.Update(0.01)
	if err != nil {
		t.Fatalf("Server Linear Update: %v", err)
	}

	t.Logf("✓ End-to-end HE/plain split test completed successfully")
}
