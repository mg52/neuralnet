package neuralnet

import "math"

// SigmoidFunc computes the Sigmoid activation function for a given input.
// Sigmoid(x) = 1 / (1 + e^(-x))
// Parameters:
// - x: Input value
// Returns:
// - The output of the Sigmoid function
func SigmoidFunc(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidPrimeFunc computes the derivative of the Sigmoid activation function.
// Sigmoid'(x) = Sigmoid(x) * (1 - Sigmoid(x))
// Parameters:
// - x: Input value
// Returns:
// - The derivative of the Sigmoid function at the input value
func SigmoidPrimeFunc(x float64) float64 {
	s := SigmoidFunc(x)
	return s * (1 - s)
}

// TanhFunc computes the Hyperbolic Tangent (Tanh) activation function for a given input.
// Tanh(x) = (e^(x) - e^(-x)) / (e^(x) + e^(-x))
// Parameters:
// - x: Input value
// Returns:
// - The output of the Tanh function
func TanhFunc(x float64) float64 {
	return math.Tanh(x)
}

// TanhPrimeFunc computes the derivative of the Hyperbolic Tangent (Tanh) activation function.
// Tanh'(x) = 1 - Tanh^2(x)
// Parameters:
// - x: Input value
// Returns:
// - The derivative of the Tanh function at the input value
func TanhPrimeFunc(x float64) float64 {
	t := math.Tanh(x)
	return 1 - t*t
}

// ReLUFunc computes the Rectified Linear Unit (ReLU) activation function for a given input.
// ReLU(x) = max(0, x)
// Parameters:
// - x: Input value
// Returns:
// - The output of the ReLU function
func ReLUFunc(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUPrimeFunc computes the derivative of the Rectified Linear Unit (ReLU) activation function.
// ReLU'(x) = 1 if x > 0, else 0
// Parameters:
// - x: Input value
// Returns:
// - The derivative of the ReLU function at the input value
func ReLUPrimeFunc(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// LinearFunc computes the Linear (identity) activation function for a given input.
// Linear(x) = x
// Parameters:
// - x: Input value
// Returns:
// - The output of the Linear function
func LinearFunc(x float64) float64 {
	return x
}

// LinearPrimeFunc computes the derivative of the Linear (identity) activation function.
// Linear'(x) = 1
// Parameters:
// - x: Input value
// Returns:
// - The derivative of the Linear function (always 1)
func LinearPrimeFunc(x float64) float64 {
	return 1
}

func softmax(z []float64) []float64 {
	maxZ := z[0]
	for _, v := range z {
		if v > maxZ {
			maxZ = v
		}
	}
	sum := 0.0
	expVals := make([]float64, len(z))
	for i, v := range z {
		expVals[i] = math.Exp(v - maxZ)
		sum += expVals[i]
	}
	for i := range expVals {
		expVals[i] /= sum
	}
	return expVals
}
