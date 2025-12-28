package main

// Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/mg52/neuralnet"
)

const (
	inputSize  = 784
	outputSize = 10
	batchSize  = 64
	epochs     = 10
)

func main() {
	net := neuralnet.NewNetwork(
		inputSize,
		[]int{64, 32},
		outputSize,
		[]string{"ReLU", "ReLU", "Softmax"},
		0.001,
		"CrossEntropy",
		"Adam",
	)

	// Load data
	base := os.Getenv("HOME")
	fmt.Println("Loading MNIST training data...")
	trainX, trainY, err := loadMNIST(base + "/Downloads/MNIST_CSV/mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Loading MNIST test data...")
	testX, testY, err := loadMNIST(base + "/Downloads/MNIST_CSV/mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	// ---------------- TRAIN ----------------
	fmt.Println("Starting Training...")
	for e := 0; e < epochs; e++ {
		start := time.Now()
		totalLoss := 0.0
		batches := 0

		for i := 0; i < len(trainX); i += batchSize {
			end := i + batchSize
			if end > len(trainX) {
				end = len(trainX)
			}

			loss := net.TrainBatch(trainX[i:end], trainY[i:end])
			totalLoss += loss
			batches++
		}

		fmt.Printf(
			"Epoch %d | loss=%.6f | time=%v\n",
			e,
			totalLoss/float64(batches),
			time.Since(start),
		)
	}

	// Save weights
	if err := net.SaveWeights("weights_mnist.json"); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Weights saved to weights_mnist.json")

	// ---------------- TEST ----------------
	fmt.Println("Testing...")

	outputs := net.PredictBatch(testX)

	correct := 0
	for i, out := range outputs {
		if argMax(out) == argMax(testY[i]) {
			correct++
		}
	}

	acc := float64(correct) / float64(len(testX)) * 100
	fmt.Printf("Test Accuracy: %.2f%% (%d/%d)\n", acc, correct, len(testX))
}

// ------------------------------------------------------------
// MNIST LOADING
// ------------------------------------------------------------

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
			return nil, nil, fmt.Errorf("expected 785 columns, got %d", len(row))
		}

		label, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, nil, err
		}

		target := make([]float64, outputSize)
		target[label] = 1.0

		input := make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			val, err := strconv.Atoi(row[j+1])
			if err != nil {
				return nil, nil, err
			}
			input[j] = float64(val) / 255.0
		}

		inputs[i] = input
		targets[i] = target
	}

	return inputs, targets, nil
}

// ------------------------------------------------------------
// HELPERS
// ------------------------------------------------------------

func argMax(values []float64) int {
	maxVal := math.Inf(-1)
	idx := 0
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			idx = i
		}
	}
	return idx
}
