package neuralnet

import (
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	// Sigmoid(0) should be 0.5
	in := 0.0
	want := 0.5
	got := SigmoidFunc(in)
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("Sigmoid(%f) = %f; want %f", in, got, want)
	}

	// SigmoidPrime(0) should be 0.25
	in = 0.0
	want = 0.25
	got = SigmoidPrimeFunc(in)
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("SigmoidPrimeFunc(%f) = %f; want %f", in, got, want)
	}

	// Test a positive input
	in = 1.0
	got = SigmoidFunc(in)
	// Approx. 1 / (1 + e^-1) ~ 0.731058
	if math.Abs(got-0.731058) > 1e-5 {
		t.Errorf("Sigmoid(1) = %f; want approx 0.731058", got)
	}

	// Test a negative input
	in = -1.0
	got = SigmoidFunc(in)
	// Approx. 1 / (1 + e^1) ~ 0.268941
	if math.Abs(got-0.268941) > 1e-5 {
		t.Errorf("Sigmoid(-1) = %f; want approx 0.268941", got)
	}
}

func TestTanh(t *testing.T) {
	// Tanh(0) = 0
	in := 0.0
	want := 0.0
	got := TanhFunc(in)
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("Tanh(%f) = %f; want %f", in, got, want)
	}

	// TanhPrimeFunc(0) = 1
	in = 0.0
	want = 1.0
	got = TanhPrimeFunc(in)
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("TanhPrimeFunc(%f) = %f; want %f", in, got, want)
	}

	// Tanh(1) ~ 0.761594
	in = 1.0
	got = TanhFunc(in)
	if math.Abs(got-0.761594) > 1e-5 {
		t.Errorf("Tanh(1) = %f; want approx 0.761594", got)
	}

	// TanhPrimeFunc(1) = 1 - Tanh^2(1) ~ 1 - 0.761594^2 ~ 0.419974
	in = 1.0
	got = TanhPrimeFunc(in)
	if math.Abs(got-0.419974) > 1e-5 {
		t.Errorf("TanhPrimeFunc(1) = %f; want approx 0.419974", got)
	}
}

func TestReLU(t *testing.T) {
	// ReLU(1) = 1
	in := 1.0
	want := 1.0
	got := ReLUFunc(in)
	if got != want {
		t.Errorf("ReLU(%f) = %f; want %f", in, got, want)
	}

	// ReLU(-1) = 0
	in = -1.0
	want = 0.0
	got = ReLUFunc(in)
	if got != want {
		t.Errorf("ReLU(%f) = %f; want %f", in, got, want)
	}

	// ReLUPrimeFunc(1) = 1
	in = 1.0
	want = 1.0
	got = ReLUPrimeFunc(in)
	if got != want {
		t.Errorf("ReLUPrimeFunc(%f) = %f; want %f", in, got, want)
	}

	// ReLUPrimeFunc(-1) = 0
	in = -1.0
	want = 0.0
	got = ReLUPrimeFunc(in)
	if got != want {
		t.Errorf("ReLUPrimeFunc(%f) = %f; want %f", in, got, want)
	}
}

func TestLinear(t *testing.T) {
	// Linear(5) = 5
	in := 5.0
	want := 5.0
	got := LinearFunc(in)
	if got != want {
		t.Errorf("Linear(%f) = %f; want %f", in, got, want)
	}

	// LinearPrimeFunc(5) = 1
	in = 5.0
	want = 1.0
	got = LinearPrimeFunc(in)
	if got != want {
		t.Errorf("LinearPrimeFunc(%f) = %f; want %f", in, got, want)
	}

	// Check for zero input
	in = 0.0
	want = 0.0
	got = LinearFunc(in)
	if got != want {
		t.Errorf("Linear(%f) = %f; want %f", in, got, want)
	}

	in = 0.0
	want = 1.0
	got = LinearPrimeFunc(in)
	if got != want {
		t.Errorf("LinearPrimeFunc(%f) = %f; want %f", in, got, want)
	}
}
