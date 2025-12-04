package split

import (
	"bytes"
	"io"
	"testing"
)

func TestProtocolRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	writer := NewProtocol(nil, &buf)

	ctBytes := []byte("test ciphertext data")
	err := writer.SendForward(1, ctBytes, 5, 1.234)
	if err != nil {
		t.Fatalf("SendForward failed: %v", err)
	}

	reader := NewProtocol(&buf, nil)
	payload, err := reader.ReceiveForward()
	if err != nil {
		t.Fatalf("ReceiveForward failed: %v", err)
	}

	if payload.BatchID != 1 {
		t.Errorf("BatchID = %d, want 1", payload.BatchID)
	}
	if payload.Level != 5 {
		t.Errorf("Level = %d, want 5", payload.Level)
	}
	if payload.ScaleFloat != 1.234 {
		t.Errorf("ScaleFloat = %f, want 1.234", payload.ScaleFloat)
	}
	if !bytes.Equal(payload.Ciphertext, ctBytes) {
		t.Errorf("Ciphertext mismatch")
	}
}

func TestProtocolGradient(t *testing.T) {
	var buf bytes.Buffer
	writer := NewProtocol(nil, &buf)

	gradBytes := []byte("gradient data")
	err := writer.SendGradient(42, gradBytes, 3, 2.5)
	if err != nil {
		t.Fatalf("SendGradient failed: %v", err)
	}

	reader := NewProtocol(&buf, nil)
	payload, err := reader.ReceiveGradient()
	if err != nil {
		t.Fatalf("ReceiveGradient failed: %v", err)
	}

	if payload.BatchID != 42 {
		t.Errorf("BatchID = %d, want 42", payload.BatchID)
	}
	if !bytes.Equal(payload.Ciphertext, gradBytes) {
		t.Errorf("Gradient data mismatch")
	}
}

func TestProtocolDone(t *testing.T) {
	var buf bytes.Buffer
	writer := NewProtocol(nil, &buf)

	err := writer.SendDone()
	if err != nil {
		t.Fatalf("SendDone failed: %v", err)
	}

	reader := NewProtocol(&buf, nil)
	_, err = reader.ReceiveForward()
	if err != io.EOF {
		t.Errorf("Expected io.EOF after done, got %v", err)
	}
}

func TestProtocolError(t *testing.T) {
	var buf bytes.Buffer
	writer := NewProtocol(nil, &buf)

	err := writer.SendError(io.ErrUnexpectedEOF)
	if err != nil {
		t.Fatalf("SendError failed: %v", err)
	}

	reader := NewProtocol(&buf, nil)
	_, err = reader.ReceiveForward()
	if err == nil {
		t.Errorf("Expected error after SendError")
	}
}

func TestMessageTypes(t *testing.T) {
	if MsgForwardInput != 0 {
		t.Errorf("MsgForwardInput = %d, want 0", MsgForwardInput)
	}
	if MsgForwardOutput != 1 {
		t.Errorf("MsgForwardOutput = %d, want 1", MsgForwardOutput)
	}
	if MsgBackwardGrad != 2 {
		t.Errorf("MsgBackwardGrad = %d, want 2", MsgBackwardGrad)
	}
	if MsgBackwardOutput != 3 {
		t.Errorf("MsgBackwardOutput = %d, want 3", MsgBackwardOutput)
	}
	if MsgUpdate != 4 {
		t.Errorf("MsgUpdate = %d, want 4", MsgUpdate)
	}
	if MsgDone != 5 {
		t.Errorf("MsgDone = %d, want 5", MsgDone)
	}
	if MsgError != 6 {
		t.Errorf("MsgError = %d, want 6", MsgError)
	}
}
