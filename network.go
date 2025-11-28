package neuralnet

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

// Activation functions
const (
	Sigmoid = "Sigmoid"
	ReLU    = "ReLU"
	Linear  = "Linear"
	Tanh    = "Tanh"
	Softmax = "Softmax"
)

// Loss functions
const (
	LossMSE          = "MSE"
	LossCrossEntropy = "CrossEntropy"
)

// Optimizers
const (
	OptimizerSGD  = "SGD"
	OptimizerAdam = "Adam"
)

// ------------------------------------------------------------
// Network Definition
// ------------------------------------------------------------

type Network struct {
	InputSize        int
	HiddenSizes      []int
	OutputSize       int
	LearningRate     float64
	Weights          [][][]float64
	Biases           [][]float64
	Activations      []func(float64) float64
	ActivationPrimes []func(float64) float64

	LossFunc       func(pred, target []float64) float64
	LossDerivative func(pred, target []float64) []float64
	LossName       string
	IsSoftmaxOut   bool

	Optimizer string
	Beta1     float64
	Beta2     float64
	Epsilon   float64

	// Adam state
	M  [][]float64
	V  [][]float64
	MW [][][]float64
	VW [][][]float64
	T  int
}

type NetworkParams struct {
	InputSize    int
	HiddenSizes  []int
	OutputSize   int
	LearningRate float64
	Weights      [][][]float64
	Biases       [][]float64
	LossName     string
}

// ------------------------------------------------------------
// Initialization
// ------------------------------------------------------------

func NewNetwork(
	inputSize int,
	hiddenSizes []int,
	outputSize int,
	activationFunctions []string,
	learningRate float64,
	lossType string,
	optimizer string,
) *Network {

	numLayers := len(hiddenSizes) + 1
	if len(activationFunctions) != numLayers {
		panic("activationFuncs must match the number of layers = len(hiddenSizes)+1")
	}

	activationFuncs := make([]func(float64) float64, numLayers)
	activationPrimeFuncs := make([]func(float64) float64, numLayers)
	isSoftmaxOut := false

	for i, name := range activationFunctions {
		switch name {
		case ReLU:
			activationFuncs[i] = ReLUFunc
			activationPrimeFuncs[i] = ReLUPrimeFunc
		case Sigmoid:
			activationFuncs[i] = SigmoidFunc
			activationPrimeFuncs[i] = SigmoidPrimeFunc
		case Tanh:
			activationFuncs[i] = TanhFunc
			activationPrimeFuncs[i] = TanhPrimeFunc
		case Linear:
			activationFuncs[i] = LinearFunc
			activationPrimeFuncs[i] = LinearPrimeFunc
		case Softmax:
			activationFuncs[i] = nil
			activationPrimeFuncs[i] = nil
			if i == numLayers-1 {
				isSoftmaxOut = true
			}
		default:
			panic("Unsupported activation function: " + name)
		}
	}

	n := &Network{
		InputSize:        inputSize,
		HiddenSizes:      hiddenSizes,
		OutputSize:       outputSize,
		LearningRate:     learningRate,
		Activations:      activationFuncs,
		ActivationPrimes: activationPrimeFuncs,
		LossName:         lossType,
		IsSoftmaxOut:     isSoftmaxOut,
		Optimizer:        optimizer,
	}

	// Select loss
	switch lossType {
	case LossMSE:
		n.LossFunc = mseLoss
		n.LossDerivative = mseLossDeriv
	case LossCrossEntropy:
		n.LossFunc = crossEntropyLoss
		n.LossDerivative = crossEntropyLossDeriv
	default:
		panic("Unsupported loss type: " + lossType)
	}

	// Layer sizes
	layerSizes := append([]int{inputSize}, hiddenSizes...)
	layerSizes = append(layerSizes, outputSize)

	// Allocate weights and biases
	n.Weights = make([][][]float64, numLayers)
	n.Biases = make([][]float64, numLayers)

	for i := 0; i < numLayers; i++ {
		n.Weights[i] = initWeights(layerSizes[i+1], layerSizes[i], activationFunctions[i])
		n.Biases[i] = randomArray(layerSizes[i+1])
	}

	// Adam initialization
	if optimizer == OptimizerAdam {
		n.Beta1 = 0.9
		n.Beta2 = 0.999
		n.Epsilon = 1e-8
		n.T = 0

		n.MW = make([][][]float64, numLayers)
		n.VW = make([][][]float64, numLayers)
		n.M = make([][]float64, numLayers)
		n.V = make([][]float64, numLayers)

		for l := 0; l < numLayers; l++ {
			n.MW[l] = zeroMatrix(len(n.Weights[l]), len(n.Weights[l][0]))
			n.VW[l] = zeroMatrix(len(n.Weights[l]), len(n.Weights[l][0]))
			n.M[l] = make([]float64, len(n.Biases[l]))
			n.V[l] = make([]float64, len(n.Biases[l]))
		}
	}

	return n
}

// ------------------------------------------------------------
// Forward Pass
// ------------------------------------------------------------

func (n *Network) Predict(input []float64) []float64 {
	a := input
	for i := 0; i < len(n.Weights); i++ {
		z := computeLayerInput(a, n.Weights[i], n.Biases[i])
		if n.IsSoftmaxOut && i == len(n.Weights)-1 {
			a = softmax(z)
		} else {
			a = applyActivation(z, n.Activations[i])
		}
	}
	return a
}

// ------------------------------------------------------------
// Training
// ------------------------------------------------------------

func (n *Network) TrainBatch(inputs, targets [][]float64) float64 {
	numLayers := len(n.Weights)
	batchSize := float64(len(inputs))

	dW := make([][][]float64, numLayers)
	dB := make([][]float64, numLayers)
	for l := 0; l < numLayers; l++ {
		dW[l] = zeroMatrix(len(n.Weights[l]), len(n.Weights[l][0]))
		dB[l] = make([]float64, len(n.Biases[l]))
	}

	totalLoss := 0.0

	for b := range inputs {
		input := inputs[b]
		target := targets[b]

		// Forward pass
		layerInputs := make([][]float64, numLayers)
		layerActivations := make([][]float64, numLayers+1)
		layerActivations[0] = input

		for i := 0; i < numLayers; i++ {
			z := computeLayerInput(layerActivations[i], n.Weights[i], n.Biases[i])
			layerInputs[i] = z

			if n.IsSoftmaxOut && i == numLayers-1 {
				layerActivations[i+1] = softmax(z)
			} else {
				layerActivations[i+1] = applyActivation(z, n.Activations[i])
			}
		}

		output := layerActivations[numLayers]
		totalLoss += n.LossFunc(output, target)
		errors := n.LossDerivative(output, target)

		// Backprop
		deltas := make([][]float64, numLayers)
		deltas[numLayers-1] = make([]float64, len(errors))

		for i := range errors {
			if n.IsSoftmaxOut && n.LossName == LossCrossEntropy {
				deltas[numLayers-1][i] = errors[i]
			} else {
				deltas[numLayers-1][i] = errors[i] * n.ActivationPrimes[numLayers-1](layerInputs[numLayers-1][i])
			}
		}

		for l := numLayers - 2; l >= 0; l-- {
			deltas[l] = make([]float64, len(n.Biases[l]))
			for i := range deltas[l] {
				sum := 0.0
				for j := 0; j < len(deltas[l+1]); j++ {
					sum += deltas[l+1][j] * n.Weights[l+1][j][i]
				}
				deltas[l][i] = sum * n.ActivationPrimes[l](layerInputs[l][i])
			}
		}

		// Accumulate gradients
		for l := 0; l < numLayers; l++ {
			for i := 0; i < len(dW[l]); i++ {
				for j := 0; j < len(dW[l][i]); j++ {
					dW[l][i][j] += deltas[l][i] * layerActivations[l][j]
				}
				dB[l][i] += deltas[l][i]
			}
		}
	}

	// Update parameters
	for l := 0; l < numLayers; l++ {
		for i := 0; i < len(dW[l]); i++ {
			for j := 0; j < len(dW[l][i]); j++ {

				gradW := dW[l][i][j] / batchSize

				// Adam
				if n.Optimizer == OptimizerAdam {
					n.T++

					n.MW[l][i][j] = n.Beta1*n.MW[l][i][j] + (1-n.Beta1)*gradW
					n.VW[l][i][j] = n.Beta2*n.VW[l][i][j] + (1-n.Beta2)*(gradW*gradW)

					mHat := n.MW[l][i][j] / (1 - math.Pow(n.Beta1, float64(n.T)))
					vHat := n.VW[l][i][j] / (1 - math.Pow(n.Beta2, float64(n.T)))

					n.Weights[l][i][j] -= n.LearningRate * mHat / (math.Sqrt(vHat) + n.Epsilon)

				} else {
					// SGD
					n.Weights[l][i][j] -= n.LearningRate * gradW
				}
			}

			gradB := dB[l][i] / batchSize

			if n.Optimizer == OptimizerAdam {
				n.M[l][i] = n.Beta1*n.M[l][i] + (1-n.Beta1)*gradB
				n.V[l][i] = n.Beta2*n.V[l][i] + (1-n.Beta2)*(gradB*gradB)

				mHatB := n.M[l][i] / (1 - math.Pow(n.Beta1, float64(n.T)))
				vHatB := n.V[l][i] / (1 - math.Pow(n.Beta2, float64(n.T)))

				n.Biases[l][i] -= n.LearningRate * mHatB / (math.Sqrt(vHatB) + n.Epsilon)
			} else {
				n.Biases[l][i] -= n.LearningRate * gradB
			}
		}
	}

	return totalLoss / batchSize
}

func (n *Network) Train(input, target []float64) float64 {
	return n.TrainBatch([][]float64{input}, [][]float64{target})
}

// ------------------------------------------------------------
// Save / Load
// ------------------------------------------------------------

func (n *Network) SaveWeights(filename string) error {
	params := NetworkParams{
		InputSize:    n.InputSize,
		HiddenSizes:  n.HiddenSizes,
		OutputSize:   n.OutputSize,
		LearningRate: n.LearningRate,
		Weights:      n.Weights,
		Biases:       n.Biases,
		LossName:     n.LossName,
	}
	data, err := json.MarshalIndent(params, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (n *Network) LoadWeights(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	var params NetworkParams
	if err := json.Unmarshal(data, &params); err != nil {
		return err
	}
	if params.InputSize != n.InputSize || params.OutputSize != n.OutputSize {
		return fmt.Errorf("mismatch network size")
	}
	n.LearningRate = params.LearningRate
	n.Weights = params.Weights
	n.Biases = params.Biases
	return nil
}

// ------------------------------------------------------------
// Loss Functions
// ------------------------------------------------------------

func mseLoss(pred, target []float64) float64 {
	sum := 0.0
	for i := range pred {
		diff := pred[i] - target[i]
		sum += diff * diff
	}
	return sum / float64(len(pred))
}

func mseLossDeriv(pred, target []float64) []float64 {
	grad := make([]float64, len(pred))
	for i := range pred {
		grad[i] = (pred[i] - target[i]) * 2.0 / float64(len(pred))
	}
	return grad
}

func crossEntropyLoss(pred, target []float64) float64 {
	eps := 1e-12
	sum := 0.0
	for i := range pred {
		p := math.Min(math.Max(pred[i], eps), 1-eps)
		sum += -(target[i]*math.Log(p) + (1-target[i])*math.Log(1-p))
	}
	return sum / float64(len(pred))
}

func crossEntropyLossDeriv(pred, target []float64) []float64 {
	grad := make([]float64, len(pred))
	for i := range pred {
		grad[i] = pred[i] - target[i]
	}
	return grad
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

func initWeights(rows, cols int, actType string) [][]float64 {
	mat := make([][]float64, rows)
	var scale float64
	switch actType {
	case ReLU:
		scale = math.Sqrt(2.0 / float64(cols))
	default:
		scale = math.Sqrt(1.0 / float64(cols))
	}
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = rand.NormFloat64() * scale
		}
	}
	return mat
}

func randomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := 0; i < size; i++ {
		arr[i] = rand.NormFloat64() * 0.1
	}
	return arr
}

func zeroMatrix(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := range mat {
		mat[i] = make([]float64, cols)
	}
	return mat
}

func computeLayerInput(a []float64, w [][]float64, b []float64) []float64 {
	z := make([]float64, len(b))
	for i := range z {
		sum := b[i]
		for j := range a {
			sum += w[i][j] * a[j]
		}
		z[i] = sum
	}
	return z
}

func applyActivation(z []float64, act func(float64) float64) []float64 {
	a := make([]float64, len(z))
	for i := range z {
		a[i] = act(z[i])
	}
	return a
}
