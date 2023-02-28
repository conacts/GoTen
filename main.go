package main

import (
	"fmt"
	"log"

	"github.com/conacts/goten/dataloader"
	"github.com/conacts/goten/nn"
)

func main() {
	// hyper parameters
	lr := 0.01
	net, err := nn.NewMLP([]int{2, 3, 4, 1})
	if err != nil {
		log.Fatalf("Failed to create new MLP: %v", err)
	}
	loss := nn.NewLoss(nn.MSE, nn.Backward)
	optimizer := nn.NewSGD(net.GetParameters(), lr)

	X, err := dataloader.LoadData("./data/xs.csv")
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	Y, err := dataloader.LoadData("./data/ys.csv")
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}

	Xs, err := dataloader.EncodeCSVToTensor(X)
	if err != nil {
		log.Fatalf("Failed to encode CSV to tensor: %v", err)
	}
	Ys, err := dataloader.EncodeCSVToTensor(Y)
	if err != nil {
		log.Fatalf("Failed to encode CSV to tensor: %v", err)
	}

	fmt.Println(net)
	for i := 0; i < 10; i++ {
		for j := 0; j < len(Xs); j++ {
			out, err := net.Forward(Xs[i])
			if err != nil {
				log.Fatalf("Forward pass failed: %v", err)
			}
			outloss, err := loss.Loss(out, Ys[i])
			if err != nil {
				log.Fatalf("Loss computation failed: %v", err)
			}
			dout, err := loss.Backward(outloss, out)
			if err != nil {
				log.Fatalf("Backward pass failed: %v", err)
			}
			net.Backward(dout)

			// Backward pass

			optimizer.Step()
		}
		net.ZeroGrad()
		fmt.Println(i)
	}
	fmt.Println(net)
}
