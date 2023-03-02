package nn

import (
	"fmt"
	"reflect"

	"math"

	"github.com/conacts/goten/engine"
)

type Loss struct {
	Criterion func(pred, y *engine.Tensor) (*engine.Tensor, error) // loss function
	Backward  func(pred, y *engine.Tensor) (*engine.Tensor, error) // backward function
	LossValue *engine.Tensor                                       // current loss value
	Gradient  *engine.Tensor                                       // gradient of loss with respect to last layer outputs
}

// EX. loss := nn.NewLoss(nn.MSE, nn.MSEBackward)
func NewLoss(lossFunc func(pred, y *engine.Tensor) (*engine.Tensor, error), backwardFunc func(pred, y *engine.Tensor) (*engine.Tensor, error)) *Loss {
	return &Loss{
		Criterion: lossFunc,
		Backward:  backwardFunc,
		LossValue: nil,
		Gradient:  nil,
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

// LogLoss computes the log loss for a binary classification problem
// yTrue and yPred are slices of true labels and predicted probabilities respectively
// They must have the same length and contain values between 0 and 1
func LogLoss(yPred, yTrue *engine.Tensor) (*engine.Tensor, error) {
	// Check if inputs are valid
	PredData := yPred.GetData()
	PredShape := yPred.GetShape()

	TrueData := yTrue.GetData()
	TrueShape := yTrue.GetShape()

	if !reflect.DeepEqual(PredShape, TrueShape) {
		return nil, fmt.Errorf("yTrue and yPred must have the same shape yPred: %v and yTrue: %v", TrueShape, PredShape)
	}
	if yTrue.GetData()[0] < 0 || yTrue.GetData()[0] > 1 {
		return nil, fmt.Errorf("yTrue must contain values between 0 and 1")
	}
	if yPred.GetData()[0] < 0 || yPred.GetData()[0] > 1 {
		return nil, fmt.Errorf("yPred must contain values between 0 and 1")
	}
	// Compute log loss
	var loss float64
	for i := range TrueData {
		loss += -TrueData[i]*math.Log(PredData[i]) - (1-TrueData[i])*math.Log(1-PredData[i])
	}
	loss /= float64(len(TrueData))
	out, _ := engine.NewTensor([]float64{loss}, []int{1, 1})
	return out, nil
}
