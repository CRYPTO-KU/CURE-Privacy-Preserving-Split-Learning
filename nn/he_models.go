package nn

import (
	"math"
	"math/rand"
	"time"

	"cure_lib/core/ckkswrapper"
	"cure_lib/nn/layers"
	"cure_lib/tensor"
	"cure_lib/utils"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// SplitHEModel represents a split HE model with server and client components
type SplitHEModel struct {
	// Server-side (HE) layers
	ServerLinear     *layers.Linear
	ServerActivation *layers.Activation

	// Client-side (plaintext) layers
	ClientLinear1    *layers.Linear
	ClientActivation *layers.Activation
	ClientLinear2    *layers.Linear

	// Training components
	LossFn       *CrossEntropyLoss
	LearningRate float64
	Stats        *utils.TimingStats
}

// NewSplitHEModel creates and initializes a split HE model
func NewSplitHEModel(arch []int, heCtx *ckkswrapper.HeContext, learningRate float64, stats *utils.TimingStats) *SplitHEModel {
	// Server-side (HE) layers
	serverLinear := layers.NewLinear(arch[0], arch[1], true, heCtx)
	serverActivation, _ := layers.NewActivation("ReLU3", true, heCtx)

	// Initialize server weights
	for i := range serverLinear.W.Data {
		serverLinear.W.Data[i] = (rand.Float64() - 0.5) * 0.1
	}
	for i := range serverLinear.B.Data {
		serverLinear.B.Data[i] = (rand.Float64() - 0.5) * 0.1
	}
	serverLinear.SyncHE()

	// Client-side (plaintext) layers
	clientLinear1 := layers.NewLinear(arch[1], arch[2], false, nil)
	clientActivation, _ := layers.NewActivation("ReLU3", false, nil)
	clientLinear2 := layers.NewLinear(arch[2], arch[3], false, nil)

	// Initialize client weights
	for i := range clientLinear1.W.Data {
		clientLinear1.W.Data[i] = (rand.Float64() - 0.5) * 0.1
	}
	for i := range clientLinear1.B.Data {
		clientLinear1.B.Data[i] = (rand.Float64() - 0.5) * 0.1
	}
	for i := range clientLinear2.W.Data {
		clientLinear2.W.Data[i] = (rand.Float64() - 0.5) * 0.1
	}
	for i := range clientLinear2.B.Data {
		clientLinear2.B.Data[i] = (rand.Float64() - 0.5) * 0.1
	}

	return &SplitHEModel{
		ServerLinear:     serverLinear,
		ServerActivation: serverActivation,
		ClientLinear1:    clientLinear1,
		ClientActivation: clientActivation,
		ClientLinear2:    clientLinear2,
		LossFn:           &CrossEntropyLoss{},
		LearningRate:     learningRate,
		Stats:            stats,
	}
}

// TrainStep performs one training step for the split HE model
func (m *SplitHEModel) TrainStep(heCtx *ckkswrapper.HeContext, inputVec, labelVec *tensor.Tensor) (float64, error) {
	// --- FORWARD PASS ---
	forwardStart := time.Now()

	// Server-side HE forward
	encryptStart := time.Now()
	ptInput := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(inputVec.Data, ptInput)
	ctInput, err := heCtx.Encryptor.EncryptNew(ptInput)
	if err != nil {
		return 0, err
	}
	m.Stats.EncryptionTime += time.Since(encryptStart)

	// Server: Linear -> Activation
	serverLinearStart := time.Now()
	ctHidden, err := m.ServerLinear.ForwardCipherMasked(ctInput)
	if err != nil {
		return 0, err
	}
	m.Stats.ServerLinearTime += time.Since(serverLinearStart)

	serverActivationStart := time.Now()
	ctHidden, err = m.ServerActivation.ForwardCipher(ctHidden)
	if err != nil {
		return 0, err
	}
	m.Stats.ServerActivationTime += time.Since(serverActivationStart)

	// Client-side forward (decrypt server output)
	decryptStart := time.Now()
	ptHidden := heCtx.Decryptor.DecryptNew(ctHidden)
	decoded := make([]complex128, m.ServerLinear.W.Shape[0])
	heCtx.Encoder.Decode(ptHidden, decoded)
	m.Stats.DecryptionTime += time.Since(decryptStart)

	clientInput := tensor.New(m.ServerLinear.W.Shape[0])
	for i := range decoded {
		clientInput.Data[i] = real(decoded[i])
	}

	// Client: Linear -> Activation -> Linear
	clientLinearStart := time.Now()
	l1Out, err := m.ClientLinear1.Forward(clientInput)
	if err != nil {
		return 0, err
	}
	m.Stats.ClientLinearTime += time.Since(clientLinearStart)

	clientActivationStart := time.Now()
	actOut, err := m.ClientActivation.Forward(l1Out)
	if err != nil {
		return 0, err
	}
	m.Stats.ClientActivationTime += time.Since(clientActivationStart)

	clientLinear2Start := time.Now()
	l2Out, err := m.ClientLinear2.Forward(actOut)
	if err != nil {
		return 0, err
	}
	m.Stats.ClientLinearTime += time.Since(clientLinear2Start)

	finalLogits := l2Out.(*tensor.Tensor)
	finalProbs := Softmax(finalLogits)

	m.Stats.ForwardPassTime += time.Since(forwardStart)

	// --- LOSS & BACKWARD ---
	backwardStart := time.Now()

	// Compute loss
	lossStart := time.Now()
	totalLoss := 0.0
	for c := 0; c < 10; c++ {
		if labelVec.Data[c] > 0 {
			p := finalProbs.Data[c]
			if p < 1e-8 {
				p = 1e-8
			}
			totalLoss -= labelVec.Data[c] * math.Log(p)
		}
	}
	m.Stats.LossComputationTime += time.Since(lossStart)

	// Backward pass (client-side)
	gradStart := m.LossFn.Backward(finalProbs, labelVec)
	gradL2, err := m.ClientLinear2.Backward(gradStart)
	if err != nil {
		return 0, err
	}
	gradAct, err := m.ClientActivation.Backward(gradL2)
	if err != nil {
		return 0, err
	}
	gradL1, err := m.ClientLinear1.Backward(gradAct)
	if err != nil {
		return 0, err
	}

	// Encrypt gradient and send to server
	encryptGradStart := time.Now()
	ptGradToServer := ckks.NewPlaintext(heCtx.Params, heCtx.Params.MaxLevel())
	heCtx.Encoder.Encode(gradL1.(*tensor.Tensor).Data, ptGradToServer)
	ctGradToServer, err := heCtx.Encryptor.EncryptNew(ptGradToServer)
	if err != nil {
		return 0, err
	}
	m.Stats.EncryptionTime += time.Since(encryptGradStart)

	// Server-side backward
	serverBackwardStart := time.Now()
	gradFromActivation, err := m.ServerActivation.BackwardHE(ctGradToServer)
	if err != nil {
		return 0, err
	}
	_, err = m.ServerLinear.BackwardHE(gradFromActivation)
	if err != nil {
		return 0, err
	}
	m.Stats.BackwardPassTime += time.Since(serverBackwardStart)

	m.Stats.BackwardPassTime += time.Since(backwardStart)

	// --- UPDATE ---
	updateStart := time.Now()
	// Update client weights
	err = m.ClientLinear1.Update(m.LearningRate)
	if err != nil {
		return 0, err
	}
	err = m.ClientLinear2.Update(m.LearningRate)
	if err != nil {
		return 0, err
	}

	// Update server weights
	err = m.ServerLinear.Update(m.LearningRate)
	if err != nil {
		return 0, err
	}
	m.Stats.UpdateTime += time.Since(updateStart)

	return totalLoss, nil
}
