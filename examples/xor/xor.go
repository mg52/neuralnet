package main

import (
	"fmt"

	"github.com/mg52/neuralnet"
)

func main() {
	net := neuralnet.NewNetwork(
		2,
		[]int{10, 5},
		1,
		[]string{"Tanh", "Tanh", "Sigmoid"},
		0.01,
		"MSE",
		"SGD",
	)

	// Example training data: XOR problem
	// Inputs and outputs
	trainX := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	trainY := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	testX := [][]float64{
		{0.001, 0},
		{0.002, 0.98},
		{1.01, 0.01},
		{0.99, 1.001},
	}
	// Train the network for a given number of epochs
	epochs := 100000
	for i := 0; i < epochs; i++ {
		err := 0.0
		for j := 0; j < len(trainX); j++ {
			err += net.Train(trainX[j], trainY[j])
		}
		if i%1000 == 0 {
			fmt.Printf("Epoch: %d, Error: %.6f\n", i, err/float64(len(trainX)))
		}
	}

	// Test the trained network
	fmt.Println("After training:")
	for i, input := range testX {
		output := net.Predict(input)
		fmt.Printf("Input: %v -> Predicted Output: %v (Target: %v)\n", input, output, trainY[i])
	}
}
