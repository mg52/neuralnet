package neuralnet

import (
	"reflect"
	"testing"
)

func TestGenerateWordToRank(t *testing.T) {
	// Test case 1: Simple frequency ranking with no stop words.
	t.Run("Basic ranking without stop words", func(t *testing.T) {
		textData := []TextData{
			{Document: []string{"apple", "banana", "apple"}, Label: 0},
			{Document: []string{"apple", "cherry", "banana"}, Label: 1},
		}
		// Frequencies:
		// apple: 3, banana: 2, cherry: 1.
		// With vocabSize = 2, we expect only the two most frequent words.
		vocabSize := 2
		stopWords := []string{}

		// Expected mapping:
		// "apple" should be rank 1 and "banana" rank 2.
		expected := map[string]int{
			"apple":  1,
			"banana": 2,
		}

		result := GenerateWordToRank(textData, vocabSize, stopWords)
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("GenerateWordToRank() = %v; want %v", result, expected)
		}
	})

	// Test case 2: Ranking with stop words filtering.
	t.Run("Ranking with stop words", func(t *testing.T) {
		textData := []TextData{
			{Document: []string{"dog", "cat", "dog"}, Label: 0},
			{Document: []string{"cat", "mouse", "dog"}, Label: 1},
		}
		// Frequencies:
		// dog: 3, cat: 2, mouse: 1.
		// Suppose we want to filter out "cat".
		vocabSize := 3
		stopWords := []string{"cat"}

		// Expected mapping should not include "cat":
		// "dog" rank 1, "mouse" rank 2.
		expected := map[string]int{
			"dog":   1,
			"mouse": 2,
		}

		result := GenerateWordToRank(textData, vocabSize, stopWords)
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("GenerateWordToRank() = %v; want %v", result, expected)
		}
	})
}

func TestOneHotBag(t *testing.T) {
	// Prepare a sample wordToRank mapping.
	wordToRank := map[string]int{
		"apple":  1,
		"banana": 2,
		"cherry": 3,
	}

	// Test case 1: All words in the doc are in the mapping and within vocabSize.
	t.Run("All words present within vocab size", func(t *testing.T) {
		doc := []string{"apple", "cherry", "banana", "apple"}
		maxWords := 4
		vocabSize := 3

		// Since we process all words (maxWords >= len(doc)),
		// each word will mark its corresponding index as 1.0.
		// Expected vector: index 0 ("apple"), index 1 ("banana"), and index 2 ("cherry") should be 1.0.
		expected := []float64{1.0, 1.0, 1.0}

		result := OneHotBag(doc, wordToRank, maxWords, vocabSize)
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("OneHotBag() = %v; want %v", result, expected)
		}
	})

	// Test case 2: maxWords less than doc length.
	t.Run("Limit doc by maxWords", func(t *testing.T) {
		doc := []string{"banana", "cherry", "apple", "banana", "cherry"}
		maxWords := 3
		vocabSize := 3

		// Only first three words are processed: "banana", "cherry", "apple".
		// Expected one-hot vector: apple (rank 1) -> index 0, banana (rank 2) -> index 1, cherry (rank 3) -> index 2.
		expected := []float64{1.0, 1.0, 1.0}

		result := OneHotBag(doc, wordToRank, maxWords, vocabSize)
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("OneHotBag() = %v; want %v", result, expected)
		}
	})

	// Test case 3: Some words in doc not in the mapping or beyond vocabSize.
	t.Run("Words not in mapping or beyond vocabSize", func(t *testing.T) {
		// Extend the mapping for this test.
		wordToRankExtended := map[string]int{
			"apple":  1,
			"banana": 2,
			"cherry": 3,
			"date":   4, // This word is in mapping but will be beyond vocabSize if vocabSize is 3.
		}
		doc := []string{"apple", "date", "banana", "kiwi"}
		maxWords := 4
		vocabSize := 3

		// "apple" (rank 1) and "banana" (rank 2) are within vocabSize.
		// "date" (rank 4) and "kiwi" (not in mapping) should be ignored.
		expected := []float64{1.0, 1.0, 0.0}

		result := OneHotBag(doc, wordToRankExtended, maxWords, vocabSize)
		if !reflect.DeepEqual(result, expected) {
			t.Errorf("OneHotBag() = %v; want %v", result, expected)
		}
	})
}

//
// Tests for the private functions
//

func TestBuildWordToRank(t *testing.T) {
	// Create a sample frequency map.
	freq := map[string]int{
		"a": 5,
		"b": 3,
		"c": 4,
		"d": 1,
	}
	// No stop words.
	stopWords := []string{}

	// Expected ranking based on frequency (highest frequency gets rank 1):
	// "a": 1, "c": 2, "b": 3, "d": 4.
	expected := map[string]int{
		"a": 1,
		"c": 2,
		"b": 3,
		"d": 4,
	}

	result := buildWordToRank(freq, stopWords)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("buildWordToRank() = %v; want %v", result, expected)
	}

	// Test with stop words filtering out "a" and "c".
	stopWords = []string{"a", "c"}
	expected = map[string]int{
		"b": 1, // becomes highest frequency among the remaining words.
		"d": 2,
	}
	result = buildWordToRank(freq, stopWords)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("buildWordToRank() with stopWords = %v; want %v", result, expected)
	}
}

func TestKeepTopKWords(t *testing.T) {
	// Prepare a sample wordToRank mapping.
	wordToRank := map[string]int{
		"apple":  1,
		"banana": 2,
		"cherry": 3,
		"date":   4,
	}

	// Test with K = 2.
	K := 2
	expected := map[string]int{
		"apple":  1,
		"banana": 2,
	}
	result := keepTopKWords(wordToRank, K)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("keepTopKWords() = %v; want %v", result, expected)
	}

	// Test with K = 4, should return the entire map.
	K = 4
	expected = wordToRank
	result = keepTopKWords(wordToRank, K)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("keepTopKWords() = %v; want %v", result, expected)
	}

	// Test with K = 0, should return an empty map.
	K = 0
	expected = map[string]int{}
	result = keepTopKWords(wordToRank, K)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("keepTopKWords() = %v; want %v", result, expected)
	}
}
