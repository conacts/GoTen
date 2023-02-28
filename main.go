package main

import (
	"github.com/conacts/goten/engine"
	"github.com/conacts/goten/nn"
	"github.com/conacts/goten/dataloader"
)

func main() {
	dataloader.LoadData("./data/xs.csv")
	engine.NewRandomTensor([]int{1, 2})
	nn.NewMLP([]int{2, 2, 1})
}
