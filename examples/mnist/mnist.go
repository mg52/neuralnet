package main

// Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/mg52/neuralnet"
)

func main() {
	epochs := 6

	net := neuralnet.NewNetwork(
		784,
		[]int{32, 16},
		10,
		[]string{"ReLU", "ReLU", "Softmax"},
		0.01,
		"CrossEntropy",
		"SGD",
	)

	// Load the MNIST training data
	fmt.Println("Loading training data...")
	base := os.Getenv("HOME")
	trainX, trainY, err := loadMNIST(base + "/Downloads/MNIST_CSV/mnist_train.csv")
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}

	// Load the MNIST test data
	fmt.Println("Loading test data...")
	testX, testY, err := loadMNIST(base + "/Downloads/MNIST_CSV/mnist_test.csv")
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	// Train the network
	fmt.Println("Starting Training...")
	for e := 0; e < epochs; e++ {
		totalErr := 0.0
		for i := 0; i < len(trainX); i++ {
			totalErr += net.Train(trainX[i], trainY[i])
		}
		fmt.Printf("Epoch: %d, Error: %.6f\n", e, totalErr/float64(len(trainX)))
	}

	// Save the trained weights
	if err := net.SaveWeights("weights.json"); err != nil {
		log.Fatalf("Error saving weights: %v", err)
	} else {
		fmt.Println("Weights saved to weights.json")
	}

	// Load Weights
	//if err := net.LoadWeights("weights.json"); err != nil {
	//	log.Fatalf("Error loading weights: %v", err)
	//} else {
	//	fmt.Println("Weights loaded from weights.json into net")
	//}

	// Test the network on the test data
	fmt.Println("Testing...")
	correct := 0
	for i := 0; i < len(testX); i++ {
		output := net.Predict(testX[i])
		predicted := argMax(output)
		actual := argMax(testY[i])
		if predicted == actual {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(testX)) * 100.0
	fmt.Printf("Test Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(testX))
}

// loadMNIST loads the MNIST data from a CSV file.
// Format: label,pixel0,pixel1,...,pixel783
// We one-hot encode the label and normalize pixels to [0,1].
func loadMNIST(filename string) ([][]float64, [][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	inputs := make([][]float64, len(records))
	targets := make([][]float64, len(records))

	for i, row := range records {
		if len(row) < 785 {
			return nil, nil, fmt.Errorf("expected at least 785 columns, got %d", len(row))
		}

		label, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, nil, fmt.Errorf("invalid label at row %d: %v", i, err)
		}

		// One-hot encode the label
		target := make([]float64, 10)
		target[label] = 1.0

		// Convert pixels to [0,1]
		input := make([]float64, 784)
		for j := 0; j < 784; j++ {
			val, err := strconv.Atoi(row[j+1])
			if err != nil {
				return nil, nil, fmt.Errorf("invalid pixel value at row %d, col %d: %v", i, j+1, err)
			}
			input[j] = float64(val) / 255.0
		}

		inputs[i] = input
		targets[i] = target
	}

	return inputs, targets, nil
}

// argMax returns the index of the maximum value in a slice.
func argMax(values []float64) int {
	maxVal := math.Inf(-1)
	maxIdx := 0
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
