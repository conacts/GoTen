package nn

import (
	"log"

	"github.com/conacts/goten/engine"
)

// Write a print function for the optimzer

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
	for i := 0; i < len(s.Parameters); i += 2 {
		// Compute the gradient of the loss with respect to the parameter
		gradW := s.Parameters[i].GetWeightGrads()
		gradB := s.Parameters[i+1].GetBiasesGrads()

		// Scale the gradient by the learning rate
		scaledGradW, err := engine.Scale(gradW, -s.LearningRate)
		if err != nil {
			log.Fatalf("failed to scale gradient: %v", err)
		}
		scaledGradB, err := engine.Scale(gradB, -s.LearningRate)
		if err != nil {
			log.Fatalf("failed to scale gradient: %v", err)
		}

		// Update the parameter using the scaled gradient
		newParamW, err := engine.Add(s.Parameters[i], scaledGradW)
		if err != nil {
			log.Fatalf("failed to update parameter: %v", err)
		}
		newParamB, err := engine.Add(s.Parameters[i+1], scaledGradB)
		if err != nil {
			log.Fatalf("failed to update parameter: %v", err)
		}
		s.Parameters[i].SetData(newParamW.GetData())
		s.Parameters[i].SetData(newParamB.GetData())
	}
}
