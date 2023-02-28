package nn

import (
	"fmt"

	"github.com/conacts/goten/engine"
)

type Loss struct {
	Loss      func(pred, y *engine.Tensor) (*engine.Tensor, error) // loss function
	Backward  func(pred, y *engine.Tensor) (*engine.Tensor, error) // backward function
	LossValue *engine.Tensor                                       // current loss value
	Gradient  *engine.Tensor                                       // gradient of loss with respect to last layer outputs
}

// EX. loss := nn.NewLoss(nn.MSE, nn.MSEBackward)
func NewLoss(lossFunc func(pred, y *engine.Tensor) (*engine.Tensor, error), backwardFunc func(pred, y *engine.Tensor) (*engine.Tensor, error)) *Loss {
	return &Loss{
		Loss:     lossFunc,
		Backward: backwardFunc,
	}
}

func Backward(preds, ans *engine.Tensor) (*engine.Tensor, error) {
	// Compute the gradient of the loss with respect to the predictions
	diff, err := engine.Sub(preds, ans)
	if err != nil {
		return nil, fmt.Errorf("failed to compute difference between preds and ans: %v", err)
	}

	// Compute the gradient of the loss with respect to the inputs (i.e., the predictions)
	scale, err := engine.Scale(diff, 2.0/float64(len(ans.GetData())))
	if err != nil {
		return nil, fmt.Errorf("failed to compute scaling factor for loss: %v", err)
	}

	return scale, nil
}

func MSE(pred, y *engine.Tensor) (*engine.Tensor, error) {
	diff, err := engine.Sub(pred, y)
	if err != nil {
		return nil, fmt.Errorf(fmt.Sprintf("failed to compute difference between yPred and yTrue: %v", err))
	}

	square, err := engine.Square(diff)
	if err != nil {
		return nil, fmt.Errorf(fmt.Sprintf("failed to compute square of difference: %v", err))
	}

	mse, err := engine.Mean(square)
	if err != nil {
		return nil, fmt.Errorf(fmt.Sprintf("failed to compute mean of square: %v", err))
	}

	return mse, nil
}
