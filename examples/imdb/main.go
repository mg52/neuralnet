package main

// Dataset: https://ai.stanford.edu/~amaas/data/sentiment/

import (
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/mg52/neuralnet"
)

const (
	maxWords  = 1000
	vocabSize = 1500
	batchSize = 64
)

var stopWords = []string{"the", "of", "in", "are", "is", "a"}

// convert slice to map for faster lookup
var stopWordSet = func() map[string]struct{} {
	m := make(map[string]struct{})
	for _, w := range stopWords {
		m[w] = struct{}{}
	}
	return m
}()

func SentenceToWords(sentence string) []string {
	re := regexp.MustCompile(`[^\w']+`)
	clean := re.ReplaceAllString(sentence, " ")
	words := strings.Fields(strings.ToLower(clean))

	var filtered []string
	for _, w := range words {
		if _, exists := stopWordSet[w]; !exists {
			filtered = append(filtered, w)
		}
	}
	return filtered
}

func main() {
	epochs := 20

	net := neuralnet.NewNetwork(
		vocabSize,
		[]int{64},
		1,
		[]string{"ReLU", "Sigmoid"},
		0.001,
		"CrossEntropy",
		"Adam",
	)

	// ---------------- TRAIN ----------------

	home := os.Getenv("HOME")
	if home == "" {
		panic("HOME not set")
	}

	posDir := home + "/Downloads/aclImdb/train/pos"
	negDir := home + "/Downloads/aclImdb/train/neg"

	imdbDataArr, err := readAllDocuments(posDir, negDir)
	if err != nil {
		panic(err)
	}

	wordToRank := neuralnet.GenerateWordToRank(imdbDataArr, vocabSize, stopWords)

	rand.Shuffle(len(imdbDataArr), func(i, j int) {
		imdbDataArr[i], imdbDataArr[j] = imdbDataArr[j], imdbDataArr[i]
	})

	fmt.Println("Training IMDB model...")

	for e := 0; e < epochs; e++ {
		start := time.Now()
		totalLoss := 0.0
		batches := 0

		for i := 0; i < len(imdbDataArr); i += batchSize {
			end := i + batchSize
			if end > len(imdbDataArr) {
				end = len(imdbDataArr)
			}

			var inputs [][]float64
			var targets [][]float64

			for _, sample := range imdbDataArr[i:end] {
				bag := neuralnet.OneHotBag(
					sample.Document,
					wordToRank,
					maxWords,
					vocabSize,
				)
				inputs = append(inputs, bag)
				targets = append(targets, []float64{float64(sample.Label)})
			}

			totalLoss += net.TrainBatch(inputs, targets)
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
	if err := net.SaveWeights("weights_IMDB.json"); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Weights saved.")

	// ---------------- TEST ----------------

	posDirTest := home + "/Downloads/aclImdb/test/pos"
	negDirTest := home + "/Downloads/aclImdb/test/neg"

	imdbTest, err := readAllDocuments(posDirTest, negDirTest)
	if err != nil {
		panic(err)
	}

	rand.Shuffle(len(imdbTest), func(i, j int) {
		imdbTest[i], imdbTest[j] = imdbTest[j], imdbTest[i]
	})

	var testInputs [][]float64
	var labels []int

	for _, sample := range imdbTest {
		bag := neuralnet.OneHotBag(
			sample.Document,
			wordToRank,
			maxWords,
			vocabSize,
		)
		testInputs = append(testInputs, bag)
		labels = append(labels, sample.Label)
	}

	outputs := net.PredictBatch(testInputs)

	correct := 0
	for i, out := range outputs {
		if labels[i] == 1 && out[0] > 0.5 {
			correct++
		}
		if labels[i] == 0 && out[0] < 0.5 {
			correct++
		}
	}

	acc := float64(correct) / float64(len(labels)) * 100
	fmt.Printf("Test Accuracy: %.2f%% (%d/%d)\n", acc, correct, len(labels))

	// ---------------- MANUAL TESTS ----------------

	testSentences := []string{
		"it was not great. the acting wasnt incredible. didnt love it.",
		"the movie was okayish but we didnt enjoy it",
		"it was brilliant, really loved it. the movie was the best.",
		"I recommend it to everybody. masterpiece.",
	}

	for _, s := range testSentences {
		words := SentenceToWords(s)
		bag := neuralnet.OneHotBag(words, wordToRank, maxWords, vocabSize)
		out := net.Predict(bag)
		fmt.Printf("'%s' => %.4f\n", s, out[0])
	}
}

// ------------------------------------------------------------
// Data loading
// ------------------------------------------------------------

func readAllDocuments(posDir, negDir string) ([]neuralnet.TextData, error) {
	var data []neuralnet.TextData

	read := func(dir string, label int) error {
		return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if !info.IsDir() && strings.HasSuffix(info.Name(), ".txt") {
				doc, err := readFileAsWords(path)
				if err != nil {
					return err
				}
				data = append(data, neuralnet.TextData{
					Document: doc,
					Label:    label,
				})
			}
			return nil
		})
	}

	if err := read(posDir, 1); err != nil {
		return nil, err
	}
	if err := read(negDir, 0); err != nil {
		return nil, err
	}

	return data, nil
}

func readFileAsWords(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	text := strings.ToLower(string(b))
	re := regexp.MustCompile(`[^a-z0-9\s]+`)
	clean := re.ReplaceAllString(text, "")
	return strings.Fields(clean), nil
}
