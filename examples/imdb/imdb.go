package main

// Dataset: https://www.kaggle.com/datasets/pranayprasad/aclimdb

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
)

var stopWords = []string{"the", "of", "in", "the", "of", "are", "is", "a"}

// convert slice to map for faster lookup
var stopWordSet = func() map[string]struct{} {
	m := make(map[string]struct{})
	for _, w := range stopWords {
		m[w] = struct{}{}
	}
	return m
}()

func SentenceToWords(sentence string) []string {
	// Remove punctuation, keep apostrophes (for contractions)
	re := regexp.MustCompile(`[^\w']+`)
	clean := re.ReplaceAllString(sentence, " ")

	// Split into words and lowercase them
	words := strings.Fields(strings.ToLower(clean))

	// Filter out stopwords
	var filtered []string
	for _, w := range words {
		if _, exists := stopWordSet[w]; !exists {
			filtered = append(filtered, w)
		}
	}

	return filtered
}

func main() {
	epochs := 50

	net := neuralnet.NewNetwork(
		vocabSize,
		[]int{32},
		1,
		[]string{"ReLU", "Sigmoid"},
		0.001,
		"CrossEntropy",
		"SGD",
	)

	// TRAIN
	home := os.Getenv("HOME")
	if home == "" {
		panic("Environment variable HOME is not set")
	}

	posDir := home + "/Downloads/aclImdb/train/pos"
	negDir := home + "/Downloads/aclImdb/train/neg"

	imdbDataArr, err := readAllDocuments(posDir, negDir)
	if err != nil {
		panic(err)
	}

	wordToRankLimited := neuralnet.GenerateWordToRank(imdbDataArr, vocabSize, stopWords)

	rand.Shuffle(len(imdbDataArr), func(i, j int) {
		imdbDataArr[i], imdbDataArr[j] = imdbDataArr[j], imdbDataArr[i]
	})

	for e := 0; e < epochs; e++ {
		totalErr := 0.0
		start := time.Now()
		for _, imdbData := range imdbDataArr {
			bag := neuralnet.OneHotBag(imdbData.Document, wordToRankLimited, maxWords, vocabSize)
			totalErr += net.Train(bag, []float64{float64(imdbData.Label)})
		}
		elapsed := time.Since(start)
		fmt.Printf("Epoch: %d, Took: %v, Error: %.6f\n", e, elapsed, totalErr/float64(vocabSize))
	}

	// Save the trained weights
	if err := net.SaveWeights("weights_IMDB.json"); err != nil {
		log.Fatalf("Error saving weights: %v", err)
	} else {
		fmt.Println("Weights saved to weights.json")
	}

	// Load Weights
	// if err := net.LoadWeights("weights_IMDB.json"); err != nil {
	// 	log.Fatalf("Error loading weights: %v", err)
	// } else {
	// 	fmt.Println("Weights loaded from weights.json into net")
	// }

	// TEST
	posDirTest := home + "/Downloads/aclImdb/test/pos"
	negDirTest := home + "/Downloads/aclImdb/test/neg"

	imdbDataArrTest, err := readAllDocuments(posDirTest, negDirTest)
	if err != nil {
		panic(err)
	}

	correct := 0

	rand.Shuffle(len(imdbDataArrTest), func(i, j int) {
		imdbDataArrTest[i], imdbDataArrTest[j] = imdbDataArrTest[j], imdbDataArrTest[i]
	})

	for i := 0; i < len(imdbDataArrTest); i++ {
		bag := neuralnet.OneHotBag(imdbDataArrTest[i].Document, wordToRankLimited, maxWords, vocabSize)
		output := net.Predict(bag)
		actual := imdbDataArrTest[i].Label
		if actual == 1 && output[0] > 0.5 {
			correct++
		}
		if actual == 0 && output[0] < 0.5 {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(imdbDataArrTest)) * 100.0
	fmt.Printf("Test Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(imdbDataArrTest))

	sentence := "it was not great. the acting wasnt incredible. didnt love it."
	words := SentenceToWords(sentence)
	testbag := neuralnet.OneHotBag(words, wordToRankLimited, maxWords, vocabSize)
	testoutput := net.Predict(testbag)
	fmt.Println("testoutput:", testoutput)

	sentence = "the movie was okeyish but my father said its not that good we didnt enjoy it really not recommended"
	words = SentenceToWords(sentence)
	testbag = neuralnet.OneHotBag(words, wordToRankLimited, maxWords, vocabSize)
	testoutput = net.Predict(testbag)
	fmt.Println("testoutput:", testoutput)

	sentence = "it was brilliant, really loved it. the movie was the best."
	words = SentenceToWords(sentence)
	testbag = neuralnet.OneHotBag(words, wordToRankLimited, maxWords, vocabSize)
	testoutput = net.Predict(testbag)
	fmt.Println("testoutput:", testoutput)

	sentence = "I recommend to everybody. its a masterpiece. very good directing and cinematography. I will watch it again."
	words = SentenceToWords(sentence)
	testbag = neuralnet.OneHotBag(words, wordToRankLimited, maxWords, vocabSize)
	testoutput = net.Predict(testbag)
	fmt.Println("testoutput:", testoutput)

}

// readAllDocuments reads *.txt files from posDir and negDir,
// tokenizes each file (removing non-alphanumeric characters).
func readAllDocuments(posDir, negDir string) ([]neuralnet.TextData, error) {
	var ImdbDataArr []neuralnet.TextData

	// Walk through train/pos
	err := filepath.Walk(posDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".txt") {
			doc, err := readFileAsWords(path)
			if err != nil {
				return err
			}
			ImdbDataArr = append(ImdbDataArr, neuralnet.TextData{
				Document: doc,
				Label:    1,
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Walk through train/neg
	err = filepath.Walk(negDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".txt") {
			doc, err := readFileAsWords(path)
			if err != nil {
				return err
			}
			ImdbDataArr = append(ImdbDataArr, neuralnet.TextData{
				Document: doc,
				Label:    0,
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return ImdbDataArr, nil
}

// readFileAsWords reads a text file into a slice of words,
// removing all non-alphanumeric characters and lowercasing.
func readFileAsWords(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	text := strings.ToLower(string(data))

	// Remove all characters that are not alphanumeric or whitespace
	re := regexp.MustCompile(`[^a-z0-9\s]+`)
	cleaned := re.ReplaceAllString(text, "")

	// Split on whitespace
	words := strings.Fields(cleaned)
	return words, nil
}
