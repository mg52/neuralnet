package main

import (
	"fmt"
	"math/rand"

	"github.com/mg52/neuralnet"
)

func main() {
	inputSize := 11 // number of features
	hidden := 32
	outputSize := 1 // click probability

	// Build NN
	net := neuralnet.NewNetwork(
		inputSize,
		[]int{hidden, hidden},
		outputSize,
		[]string{"ReLU", "ReLU", "Sigmoid"},
		0.001,
		"CrossEntropy",
		"Adam",
	)

	// Generate synthetic training data
	trainX, trainY := generateTrainingData(5000)

	fmt.Println("Training recommendation model...")

	epochs := 10
	for e := 0; e < epochs; e++ {
		totalLoss := 0.0
		for i := 0; i < len(trainX); i++ {
			totalLoss += net.Train(trainX[i], trainY[i])
		}
		fmt.Printf("Epoch %d: Loss = %.4f\n", e, totalLoss/float64(len(trainX)))
	}

	fmt.Println("\nTesting predictions...")
	testModel(net)
}

// -------------------------
// SYNTHETIC DATA GENERATION
// -------------------------
func generateTrainingData(n int) ([][]float64, [][]float64) {
	X := make([][]float64, n)
	Y := make([][]float64, n)

	for i := 0; i < n; i++ {
		userAge := rand.Float64() // 0–1 normalized
		userActivity := rand.Float64()

		userPref := rand.Intn(3) // 0,1,2
		userPref1 := boolFloat(userPref == 0)
		userPref2 := boolFloat(userPref == 1)
		userPref3 := boolFloat(userPref == 2)

		itemCat := rand.Intn(3)
		itemCat1 := boolFloat(itemCat == 0)
		itemCat2 := boolFloat(itemCat == 1)
		itemCat3 := boolFloat(itemCat == 2)

		itemPopularity := rand.Float64()
		itemPrice := rand.Float64()

		// Similarity feature (this correlates behavior)
		similarity := 1.0
		if userPref != itemCat {
			similarity = 0.1
		}

		// Build feature vector
		features := []float64{
			userAge,
			userActivity,
			userPref1, userPref2, userPref3,

			itemPopularity,
			itemPrice,
			itemCat1, itemCat2, itemCat3,

			similarity,
		}

		// -----------------------------
		// Synthetic click formula
		// -----------------------------
		clickProb := 0.0

		// User prefers items in their preferred category
		if userPref == itemCat {
			clickProb += 0.5
		}

		// Popular items more likely to be clicked
		clickProb += 0.3 * itemPopularity

		// Expensive items less likely to be clicked
		clickProb -= 0.2 * itemPrice

		// High activity users click more
		clickProb += 0.2 * userActivity

		// Similarity is big factor
		clickProb += 0.3 * similarity

		// Clamp and treat as probability
		if clickProb < 0 {
			clickProb = 0
		}
		if clickProb > 1 {
			clickProb = 1
		}

		click := 0.0
		if rand.Float64() < clickProb {
			click = 1.0
		}

		X[i] = features
		Y[i] = []float64{click}
	}

	return X, Y
}

func boolFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// -------------------------
// TESTING
// -------------------------
func testModel(net *neuralnet.Network) {
	user := []float64{
		0.4,     // age
		0.9,     // active user
		1, 0, 0, // user prefers category 0
	}

	// Try 3 items with different categories & popularity
	items := [][]float64{
		{0.9, 0.2, 1, 0, 0, 1},   // popular, cheap, correct category
		{0.7, 0.7, 0, 1, 0, 0.1}, // wrong category
		{0.2, 0.3, 1, 0, 0, 0.8}, // low popularity
	}

	for i, it := range items {
		features := append(user, it...)
		score := net.Predict(features)[0]
		fmt.Printf("Item %d predicted click score = %.3f\n", i, score)
	}
}
