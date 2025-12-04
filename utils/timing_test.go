package utils

import (
	"math"
	"testing"
	"time"
)

func TestDurationUS(t *testing.T) {
	d := 1234*time.Microsecond + 567*time.Nanosecond
	got := DurationUS(d)
	if math.Abs(got-1234.567) > 0.001 {
		t.Fatalf("want 1234.567µs, got %.3f", got)
	}
}
