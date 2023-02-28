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

// Parameters returns a slice of all the parameters in the MLP
func (m *MLP) GetParameters() []*engine.Tensor {
	params := make([]*engine.Tensor, 0)
	for _, l := range m.layers {
		params = append(params, l.w)
		params = append(params, l.b)
	}
	return params
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
	w    *engine.Tensor
	b    *engine.Tensor
	lin  int
	lout int
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
		w:    w,
		b:    b,
		lin:  lin,
		lout: lout,
	}, nil
}

// Input should be of the shape [lin, lout]
func (l *LinearLayer) Forward(x *engine.Tensor) (*engine.Tensor, error) {
	if x.GetShape()[1] != l.lin {
		return nil, fmt.Errorf("input tensor has invalid shape, %v compared to %d", x.GetShape(), l.lin)
	}

	fmt.Printf("x: %v, l.w: %v, l.b %v\n", x.GetShape(), l.w.GetShape(), l.b.GetShape())
	// Compute linear transformation
	/*
		wt, err := engine.Transpose(l.w)
		if err != nil {
			return nil, err
		}
	*/
	z, err := engine.Dot(x, l.w)
	if err != nil {
		return nil, err
	}
	/*
		bt, err := engine.Transpose(l.b)
		if err != nil {
			return nil, err
		}
	*/
	z, err = engine.Add(z, l.b)
	if err != nil {
		return nil, err
	}

	return z, nil
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

func (l *LinearLayer) GetParameters() []*engine.Tensor {
	return []*engine.Tensor{l.w, l.b}
}
