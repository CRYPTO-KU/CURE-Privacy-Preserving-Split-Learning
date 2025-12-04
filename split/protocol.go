// Package split provides utilities for split learning between server and client
package split

import (
	"encoding/gob"
	"fmt"
	"io"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func init() {
	// Register types for gob encoding
	gob.Register(&rlwe.Ciphertext{})
	gob.Register(&rlwe.Plaintext{})
	gob.Register(ForwardPayload{})
	gob.Register(GradientPayload{})
}

// MessageType defines message types for split learning protocol
type MessageType int

const (
	MsgForwardInput MessageType = iota
	MsgForwardOutput
	MsgBackwardGrad
	MsgBackwardOutput
	MsgUpdate
	MsgDone
	MsgError
)

// Message represents a message in the split learning protocol
type Message struct {
	Type    MessageType
	Payload interface{}
}

// ForwardPayload contains encrypted forward pass data
type ForwardPayload struct {
	BatchID    int
	Ciphertext []byte // serialized ciphertext
	Level      int
	ScaleFloat float64
}

// GradientPayload contains encrypted gradient data
type GradientPayload struct {
	BatchID    int
	Ciphertext []byte
	Level      int
	ScaleFloat float64
}

// Protocol handles split learning communication
type Protocol struct {
	encoder *gob.Encoder
	decoder *gob.Decoder
}

// NewProtocol creates a new protocol handler
func NewProtocol(r io.Reader, w io.Writer) *Protocol {
	return &Protocol{
		encoder: gob.NewEncoder(w),
		decoder: gob.NewDecoder(r),
	}
}

// Send sends a message
func (p *Protocol) Send(msg *Message) error {
	return p.encoder.Encode(msg)
}

// Receive receives a message
func (p *Protocol) Receive() (*Message, error) {
	var msg Message
	if err := p.decoder.Decode(&msg); err != nil {
		return nil, err
	}
	return &msg, nil
}

// SendForward sends a forward pass ciphertext
func (p *Protocol) SendForward(batchID int, ctBytes []byte, level int, scale float64) error {
	return p.Send(&Message{
		Type: MsgForwardInput,
		Payload: ForwardPayload{
			BatchID:    batchID,
			Ciphertext: ctBytes,
			Level:      level,
			ScaleFloat: scale,
		},
	})
}

// SendGradient sends a gradient ciphertext
func (p *Protocol) SendGradient(batchID int, ctBytes []byte, level int, scale float64) error {
	return p.Send(&Message{
		Type: MsgBackwardGrad,
		Payload: GradientPayload{
			BatchID:    batchID,
			Ciphertext: ctBytes,
			Level:      level,
			ScaleFloat: scale,
		},
	})
}

// SendDone signals completion
func (p *Protocol) SendDone() error {
	return p.Send(&Message{Type: MsgDone})
}

// SendError sends an error message
func (p *Protocol) SendError(err error) error {
	return p.Send(&Message{
		Type:    MsgError,
		Payload: err.Error(),
	})
}

// ReceiveForward receives a forward pass payload
func (p *Protocol) ReceiveForward() (*ForwardPayload, error) {
	msg, err := p.Receive()
	if err != nil {
		return nil, err
	}
	if msg.Type == MsgError {
		return nil, fmt.Errorf("remote error: %v", msg.Payload)
	}
	if msg.Type == MsgDone {
		return nil, io.EOF
	}
	if msg.Type != MsgForwardInput && msg.Type != MsgForwardOutput {
		return nil, fmt.Errorf("expected forward message, got %d", msg.Type)
	}
	payload, ok := msg.Payload.(ForwardPayload)
	if !ok {
		return nil, fmt.Errorf("invalid forward payload type")
	}
	return &payload, nil
}

// ReceiveGradient receives a gradient payload
func (p *Protocol) ReceiveGradient() (*GradientPayload, error) {
	msg, err := p.Receive()
	if err != nil {
		return nil, err
	}
	if msg.Type == MsgError {
		return nil, fmt.Errorf("remote error: %v", msg.Payload)
	}
	if msg.Type == MsgDone {
		return nil, io.EOF
	}
	if msg.Type != MsgBackwardGrad && msg.Type != MsgBackwardOutput {
		return nil, fmt.Errorf("expected gradient message, got %d", msg.Type)
	}
	payload, ok := msg.Payload.(GradientPayload)
	if !ok {
		return nil, fmt.Errorf("invalid gradient payload type")
	}
	return &payload, nil
}
