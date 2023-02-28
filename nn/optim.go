package nn

import (
	"log"

	"github.com/conacts/goten/engine"
)

type SGD struct {
	Parameters   []*engine.Tensor
	LearningRate float64
}

func NewSGD(params []*engine.Tensor, learningRate float64) *SGD {
	return &SGD{
		Parameters:   params,
		LearningRate: learningRate,
	}
}

func (s *SGD) Step() {
	// Update the parameters of each layer
	for _, p := range s.Parameters {
		// Compute the gradient of the loss with respect to the parameter
		grad := p.GetWeights()

		// Scale the gradient by the learning rate
		scaledGrad, err := engine.Scale(grad, -s.LearningRate)
		if err != nil {
			log.Fatalf("failed to scale gradient: %v", err)
		}

		// Update the parameter using the scaled gradient
		newParam, err := engine.Add(p, scaledGrad)
		if err != nil {
			log.Fatalf("failed to update parameter: %v", err)
		}

		p.SetData(newParam.GetData())
	}
}
