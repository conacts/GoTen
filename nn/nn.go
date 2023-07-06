package nn

import (
	"fmt"
	"strings"

	"github.com/conacts/goten/engine"
)

type MLP struct {
	layers []*LinearLayer
}

func NewMLP(layerSizes []int) (*MLP, error) {
	if len(layerSizes) < 2 {
		return nil, fmt.Errorf("invalid number of layer sizes")
	}

	// create linear layers
	layers := make([]*LinearLayer, len(layerSizes)-1)
	for i := 0; i < len(layerSizes)-1; i++ {
		ll, err := NewLinearLayer(layerSizes[i], layerSizes[i+1])
		if err != nil {
			return nil, fmt.Errorf("failed to create linear layer: %v", err)
		}
		layers[i] = ll
	}

	return &MLP{layers}, nil
}

// Forward runs the forward pass through the MLP
/*
func (m *MLP) Forward(x *engine.Tensor) (*engine.Tensor, error) {
	out := x
	var err error
	for _, l := range m.layers {
		out, err = l.Forward(out)
		if err != nil {
			return nil, fmt.Errorf("failed to run forward pass through linear layer: %v", err)
		}
		out, err = engine.Sigmoid(out)
		if err != nil {
			return nil, fmt.Errorf("failed to apply ReLU activation: %v", err)
		}
	}
	return out, nil
}
*/

// Forward runs the forward pass through the MLP
func (m *MLP) Forward(x *engine.Tensor) (*engine.Tensor, error) {
	out := x
	var err error
	for _, l := range m.layers {
		out, err = l.Forward(out)
		if err != nil {
			return nil, fmt.Errorf("failed to run forward pass through linear layer: %v", err)
		}
		out, err = engine.Relu(out)
		if err != nil {
			return nil, fmt.Errorf("failed to apply ReLU activation: %v", err)
		}
	}
	return out, nil
}

func (m *MLP) GetLayers() []*LinearLayer {
	return m.layers
}

// Parameters returns a slice of all the parameters in the MLP
func (m *MLP) GetParameters() []*engine.Tensor {
	params := make([]*engine.Tensor, 0)
	for _, l := range m.layers {
		params = append(params, l.w)
		params = append(params, l.b)
	}
	return params
}

func (m *MLP) Backward(dout *engine.Tensor) error {
	// Iterate through the layers in reverse order
	for i := len(m.layers) - 1; i >= 0; i-- {
		// layer := m.layers[i]

		// Backpropagate through the linear layer
		dz, err := m.layers[i].Backward(dout)
		if err != nil {
			return fmt.Errorf("error in layer %d backward: %v", i+1, err)
		}

		// Set the output gradient to the input gradient of the previous layer
		dout = dz
	}

	return nil
}

func (m *MLP) ZeroGrad() {
	for _, layer := range m.layers {
		layer.ZeroGrad()
	}
}

func (m *MLP) String() string {
	var sb strings.Builder

	// Add information about each layer to the string
	for i, layer := range m.layers {
		fmt.Fprintf(&sb, "Layer %d:\n", i+1)
		fmt.Fprintf(&sb, "  Type: %T\n", layer)
		fmt.Fprintf(&sb, "  Input shape: %v\n", layer.lin)
		fmt.Fprintf(&sb, "  Output shape: %v\n", layer.lout)
		fmt.Fprintf(&sb, "  Parameters:\n")
		for j, param := range layer.GetParameters() {
			if j == 0 {
				fmt.Fprintf(&sb, "    %s: %v\n", "Weight: ", param)
			} else {
				fmt.Fprintf(&sb, "    %s: %v\n", "Biases: ", param)
			}
		}
	}

	return sb.String()
}

type LinearLayer struct {
	w        *engine.Tensor
	b        *engine.Tensor
	lin      int
	lout     int
	intensor *engine.Tensor
}

func (l *LinearLayer) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Linear layer: %d -> %d\n", l.lin, l.lout))
	sb.WriteString(fmt.Sprintf("Weights:\n%v\n", l.w))
	sb.WriteString(fmt.Sprintf("Weight Gradients:\n%v\n", l.w.GetWeightGrads()))
	sb.WriteString(fmt.Sprintf("Biases:\n%v\n", l.b))
	sb.WriteString(fmt.Sprintf("Bias Gradients:\n%v\n", l.b.GetBiasesGrads()))
	sb.WriteString(fmt.Sprintf("Input:\n%v\n", l.intensor))
	sb.WriteString(fmt.Sprintf("Input Gradient:\n%v\n", l.intensor.GetInputGrads()))
	return sb.String()
}

func NewLinearLayer(lin, lout int) (*LinearLayer, error) {
	w, err1 := engine.NewRandomTensor([]int{lin, lout})
	if err1 != nil {
		return nil, fmt.Errorf("failed to create weight tensor: %v", err1)
	}

	b, err2 := engine.NewRandomTensor([]int{1, lout})
	if err2 != nil {
		return nil, fmt.Errorf("failed to create bias tensor: %v", err2)
	}

	return &LinearLayer{
		w:        w,
		b:        b,
		lin:      lin,
		lout:     lout,
		intensor: nil,
	}, nil
}

/*
// Input should be of the shape [lin, lout]
func (l *LinearLayer) Forward(x *engine.Tensor) (*engine.Tensor, error) {
	if x.GetShape()[1] != l.lin {
		return nil, fmt.Errorf("input tensor has invalid shape, %v compared to %d", x.GetShape(), l.lin)
	}

	l.intensor = x
	// Compute linear transformation
		wt, err := engine.Transpose(l.w)
		if err != nil {
			return nil, err
		}
	z, err := engine.Dot(x, l.w)
	if err != nil {
		return nil, err
	}
		bt, err := engine.Transpose(l.b)
		if err != nil {
			return nil, err
		}
	z, err = engine.Add(z, l.b)
	if err != nil {
		return nil, err
	}

	return z, nil
}
*/

// Input should be of the shape [batch, lin]
func (l *LinearLayer) Forward(x *engine.Tensor) (*engine.Tensor, error) {
	/*
	if x.GetShape()[1] != l.lin {
		return nil, fmt.Errorf("input tensor has invalid shape, %v compared to %d", x.GetShape(), l.lin)
	}
	*/

	l.intensor = x
	// Compute linear transformation
	z, err := engine.Dot(x, l.w)
	if err != nil {
		return nil, err
	}

	// Add biases to the linear transformation
	b, err := engine.NewTensor(l.b.GetData(), []int{x.GetShape()[0], l.lout})
	if err != nil {
		return nil, err
	}
	z, err = engine.Add(z, b)
	if err != nil {
		return nil, err
	}

	return z, nil
}

func (l *LinearLayer) Backward(dout *engine.Tensor) (*engine.Tensor, error) {
	// Check if the shape of input tensor is valid
	if dout.GetShape()[1] != l.lout {
		return nil, fmt.Errorf("invalid shape for input tensor, expected (%d, %d), got %v", dout.GetShape()[0], l.lout, dout.GetShape())
	}

	// Compute gradients of weights and biases
	// Transpose input tensor
	inputT, err := engine.Transpose(l.intensor)
	if err != nil {
		return nil, err
	}
	// Compute gradient of weights
	dw, err := engine.Dot(inputT, dout)
	if err != nil {
		return nil, err
	}

	// Save the gradients in the layer
	// Create new Tensor for weight gradient and set it
	weightGrads, err := engine.NewTensor(dw.GetData(), dw.GetShape())
	if err != nil {
		return nil, err
	}
	l.w.SetWeightGrads(weightGrads)

	// Compute gradient of biases
	// Sum along axis 0 of dout to obtain the gradients for each example
	db, err := engine.SumCols(dout)
	if err != nil {
		return nil, err
	}
	// Reshape bias gradient tensor to match the shape of the bias tensor
	biasGrads, err := engine.NewTensor(db.GetData(), []int{1, db.GetShape()[0]})
	if err != nil {
		return nil, err
	}
	l.b.SetBiasGrads(biasGrads)

	// Propagate the gradient to the input
	// Transpose weight tensor
	weightT, err := engine.Transpose(l.w)
	if err != nil {
		return nil, err
	}
	// Compute gradient of input
	doutWt, err := engine.Dot(dout, weightT)
	if err != nil {
		return nil, err
	}
	doutWt.SetWeightGrads(dout)

	return doutWt, nil
}

func (l *LinearLayer) ZeroGrad() {
	w, _ := engine.NewZeroTensor(l.w.GetShape())
	b, _ := engine.NewZeroTensor(l.b.GetShape())

	l.w.SetWeightGrads(w)
	l.b.SetBiasGrads(b)
}

func (ll *LinearLayer) GetWeights() *engine.Tensor {
	return ll.w
}

func (ll *LinearLayer) SetWeights(w *engine.Tensor) {
	ll.w = w
}

func (ll *LinearLayer) GetBiases() *engine.Tensor {
	return ll.b
}

func (ll *LinearLayer) SetBiases(b *engine.Tensor) {
	ll.b = b
}

func (ll *LinearLayer) GetIntensor() *engine.Tensor {
	return ll.intensor
}

func (l *LinearLayer) GetParameters() []*engine.Tensor {
	return []*engine.Tensor{l.w, l.b}
}
