# Goten

Goten is a tensor library written in Go with a neural network library built on top. It is designed to be easy to use, performant, and suitable for a variety of machine learning and scientific computing tasks.

## Installation

To install Goten, simply run:

```go
go get github.com/conacts/goten
```

## Usage

Here is an example usage of Goten:

```go
package main

import (
 "fmt"
 "github.com/conacts/goten/engine"
)

func main() {
 // create a tensor with shape (2, 3) and initialize it with random values
 t := engine.NewRandomTensor([]int{2, 3})

 // print the tensor's values and shape
 fmt.Println("Tensor values:", t.GetData())
 fmt.Println("Tensor shape:", t.GetShape())

 // perform element-wise multiplication with a tensor of the same shape
 u := engine.NewTensor([]float64{1,1,1,1,1,1}, []int{2, 3})
 v := Mul(t, u)

 // print the result
 fmt.Println("Result:", v.GetData())
}
```

## Contributing

If you would like to contribute to goten, please open an issue or a pull request on the GitHub repository.

## To Do

1. Finish testing for new functions
2. Implement proper backpropogation algorithms for Loss and Layers
