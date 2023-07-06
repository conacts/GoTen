package test

import (
	"reflect"
	"testing"

	"github.com/conacts/goten/engine"
	"github.com/conacts/goten/nn"
)

func TestCreateLinearLayer(t *testing.T) {
	tests := []struct {
		name    string
		lin     int
		lout    int
		wantErr bool
	}{
		{
			name:    "valid input",
			lin:     3,
			lout:    2,
			wantErr: false,
		},
		{
			name:    "invalid input",
			lin:     0,
			lout:    0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := nn.NewLinearLayer(tt.lin, tt.lout)

			if (err != nil) != tt.wantErr {
				t.Errorf("CreAteLinearLayer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if got != nil && got.GetWeights().GetShape()[0] != tt.lin {
				t.Errorf("CreateLinearLayer() expected shape = [%v, %v], got = %v", tt.lin, tt.lout, got.GetWeights().GetShape())
			}

			if got != nil && got.GetWeights().GetShape()[1] != tt.lout {
				t.Errorf("CreateLinearLayer() expected shape = [%v, %v], got = %v", tt.lin, tt.lout, got.GetWeights().GetShape())
			}

			if got != nil && got.GetBiases().GetShape()[0] != 1 {
				t.Errorf("CreateLinearLayer() expected bias shape = [1, %v], got = %v", tt.lout, got.GetBiases().GetShape())
			}

			if got != nil && got.GetBiases().GetShape()[1] != tt.lout {
				t.Errorf("CreateLinearLayer() expected bias shape = [1, %v], got = %v", tt.lout, got.GetBiases().GetShape())
			}
		})
	}
}

func TestLinearLayer_GetSetWeights(t *testing.T) {
	lin, lout := 3, 2
	ll, err := nn.NewLinearLayer(lin, lout)
	if err != nil {
		t.Errorf("Error creating LinearLayer: %v", err)
	}

	// Test GetWeights method
	w := ll.GetWeights()
	if w == nil {
		t.Errorf("LinearLayer weights tensor is nil")
	}
	if !reflect.DeepEqual(w.GetShape(), []int{lin, lout}) {
		t.Errorf("LinearLayer weights tensor has incorrect shape")
	}

	// Test SetWeights method
	newW, err := engine.NewRandomTensor([]int{lin, lout})
	if err != nil {
		t.Errorf("Error creating new weights tensor: %v", err)
	}
	ll.SetWeights(newW)
	if !reflect.DeepEqual(newW, ll.GetWeights()) {
		t.Errorf("LinearLayer weights tensor was not set correctly")
	}
}

func TestLinearLayer_GetSetBiases(t *testing.T) {
	lin, lout := 3, 2
	ll, err := nn.NewLinearLayer(lin, lout)
	if err != nil {
		t.Errorf("Error creating LinearLayer: %v", err)
	}

	// Test GetBiases method
	b := ll.GetBiases()
	if b == nil {
		t.Errorf("LinearLayer biases tensor is nil")
	}
	if !reflect.DeepEqual(b.GetShape(), []int{1, lout}) {
		t.Errorf("LinearLayer biases tensor has incorrect shape")
	}

	// Test SetBiases method
	newB, err := engine.NewRandomTensor([]int{1, lout})
	if err != nil {
		t.Errorf("Error creating new biases tensor: %v", err)
	}
	ll.SetBiases(newB)
	if !reflect.DeepEqual(newB, ll.GetBiases()) {
		t.Errorf("LinearLayer biases tensor was not set correctly")
	}
}

// WRITE FOR REAL PREDICTED OUTPUT
/*
func TestLinearLayer_Forward(t *testing.T) {
	lin, lout := 3, 2
	ll, err := nn.NewLinearLayer(lin, lout)
	if err != nil {
		t.Fatalf("failed to create linear layer: %v", err)
	}

	// Create input tensor
	x, err := engine.NewRandomTensor([]int{1, lin})
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}
	xflat, err := engine.Flatten(x)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	// Compute forward pass
	out, err := ll.Forward(xflat)
	if err != nil {
		t.Fatalf("failed to compute forward pass: %v", err)
	}

	// Check output tensor shape
	expectedShape := []int{lout, 1}
	if !reflect.DeepEqual(out.GetShape(), expectedShape) {
		t.Errorf("unexpected output tensor shape: got %v, want %v", out.GetShape(), expectedShape)
	}

	rand.Seed(1)
	tens, _ := engine.NewRandomTensor([]int{2, 1})
	l, _ := nn.NewLinearLayer(2, 2)
	v, _ := l.Forward(tens)
	expectedout, _ := engine.NewTensor([]float64{-0.932625821077598, -0.38384784914080966}, []int{2, 1})
	if !reflect.DeepEqual(expectedout, v) {
		t.Errorf("unexpected output: got %v, want %v", v, expectedout)
	}
}
*/
