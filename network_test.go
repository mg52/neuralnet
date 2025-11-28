package neuralnet

import (
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

// ---------------------------------------------------------
// Activation Function Tests
// ---------------------------------------------------------

func TestActivationFunctions(t *testing.T) {
	assert.InDelta(t, 0.5, SigmoidFunc(0), 0.001)
	assert.InDelta(t, 0.0, ReLUFunc(-2), 0.001)
	assert.InDelta(t, 2.0, ReLUFunc(2), 0.001)
	assert.InDelta(t, math.Tanh(1.0), TanhFunc(1.0), 0.0001)
	assert.InDelta(t, 5.0, LinearFunc(5.0), 0.001)
}

// ---------------------------------------------------------
// Weight Initialization
// ---------------------------------------------------------

func TestInitWeightsShape(t *testing.T) {
	w := initWeights(4, 3, ReLU)
	assert.Equal(t, 4, len(w))
	assert.Equal(t, 3, len(w[0]))
}

// ---------------------------------------------------------
// Loss Function Tests
// ---------------------------------------------------------

func TestMSELoss(t *testing.T) {
	p := []float64{0.1, 0.9}
	tg := []float64{0, 1}
	l := mseLoss(p, tg)
	assert.True(t, l > 0)
}

func TestCrossEntropyLoss(t *testing.T) {
	p := []float64{0.2}
	tg := []float64{1.0}
	l := crossEntropyLoss(p, tg)
	assert.True(t, l > 0)
}

// ---------------------------------------------------------
// Softmax Test
// ---------------------------------------------------------

func TestSoftmax(t *testing.T) {
	z := []float64{1.0, 2.0, 3.0}
	s := softmax(z)

	sum := 0.0
	for _, v := range s {
		sum += v
	}

	assert.InDelta(t, 1.0, sum, 0.0001)
	assert.True(t, s[2] > s[1] && s[1] > s[0])
}

// ---------------------------------------------------------
// Network Creation
// ---------------------------------------------------------

func TestNewNetwork(t *testing.T) {
	n := NewNetwork(
		4,
		[]int{5},
		3,
		[]string{ReLU, Softmax},
		0.01,
		LossCrossEntropy,
		OptimizerSGD,
	)

	assert.Equal(t, 4, n.InputSize)
	assert.Equal(t, 3, n.OutputSize)
	assert.Equal(t, 2, len(n.Weights))
	assert.True(t, n.IsSoftmaxOut)
}

// ---------------------------------------------------------
// Forward Pass
// ---------------------------------------------------------

func TestPredictShape(t *testing.T) {
	n := NewNetwork(
		3,
		[]int{4},
		2,
		[]string{ReLU, Softmax},
		0.01,
		LossCrossEntropy,
		OptimizerSGD,
	)

	out := n.Predict([]float64{0.2, 0.1, 0.9})
	assert.Equal(t, 2, len(out))
}

// ---------------------------------------------------------
// Save & Load
// ---------------------------------------------------------

func TestSaveLoad(t *testing.T) {
	n1 := NewNetwork(
		3,
		[]int{4},
		1,
		[]string{Tanh, Sigmoid},
		0.05,
		LossMSE,
		OptimizerSGD,
	)

	filename := "test_weights.json"
	err := n1.SaveWeights(filename)
	assert.NoError(t, err)

	defer os.Remove(filename)

	n2 := NewNetwork(
		3,
		[]int{4},
		1,
		[]string{Tanh, Sigmoid},
		0.05,
		LossMSE,
		OptimizerSGD,
	)

	err = n2.LoadWeights(filename)
	assert.NoError(t, err)

	assert.Equal(t, n1.Weights, n2.Weights)
	assert.Equal(t, n1.Biases, n2.Biases)
}

// ---------------------------------------------------------
// Simple Training - XOR
// ---------------------------------------------------------

func TestTrainXOR_SGD(t *testing.T) {
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	n := NewNetwork(
		2,
		[]int{4},
		1,
		[]string{Tanh, Sigmoid},
		0.1,
		LossMSE,
		OptimizerSGD,
	)

	for epoch := 0; epoch < 7000; epoch++ {
		for i := range inputs {
			n.Train(inputs[i], targets[i])
		}
	}

	for i := range inputs {
		out := n.Predict(inputs[i])[0]
		exp := targets[i][0]
		assert.InDelta(t, exp, out, 0.3)
	}
}

// ---------------------------------------------------------
// Adam Optimizer Test
// ---------------------------------------------------------

func TestTrainXOR_Adam(t *testing.T) {
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	n := NewNetwork(
		2,
		[]int{4},
		1,
		[]string{Tanh, Sigmoid},
		0.01,
		LossMSE,
		OptimizerAdam,
	)

	for epoch := 0; epoch < 2000; epoch++ {
		for i := range inputs {
			n.Train(inputs[i], targets[i])
		}
	}

	for i := range inputs {
		out := n.Predict(inputs[i])[0]
		exp := targets[i][0]
		assert.InDelta(t, exp, out, 0.3)
	}
}

// ---------------------------------------------------------
// CrossEntropy + Softmax Test (Multiclass)
// ---------------------------------------------------------

func TestSoftmaxCrossEntropyTraining(t *testing.T) {
	inputs := [][]float64{
		{1, 0},
		{0, 1},
	}
	targets := [][]float64{
		{1, 0},
		{0, 1},
	}

	n := NewNetwork(
		2,
		[]int{4},
		2,
		[]string{ReLU, Softmax},
		0.1,
		LossCrossEntropy,
		OptimizerSGD,
	)

	for epoch := 0; epoch < 2000; epoch++ {
		for i := range inputs {
			n.Train(inputs[i], targets[i])
		}
	}

	out1 := n.Predict([]float64{1, 0})
	out2 := n.Predict([]float64{0, 1})

	assert.True(t, out1[0] > out1[1])
	assert.True(t, out2[1] > out2[0])
}
