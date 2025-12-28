# Neural Network

Simple neural network implementation in Golang.

This project contains:
- A fully connected feed-forward neural network
- Support for multiple activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax)
- Multiple loss functions (MSE, Cross-Entropy)
- Two optimization options (SGD and Adam)
- Saving and loading model weights (JSON)
- Text utilities for bag-of-words and negation handling
- Example scripts for MNIST and IMDB sentiment classification

## Installation

```sh
go get github.com/mg52/neuralnet
```

## Example

```go
net := neuralnet.NewNetwork(
    784,
    []int{32, 16},
    10,
    []string{"ReLU", "ReLU", "Softmax"},
    0.01,
    "CrossEntropy",
    "Adam",
)

output := net.Predict(inputVector)
```

### Test

```shell
go test -coverprofile=coverage.out -v -race ./...
```

## License

MIT © mg52
