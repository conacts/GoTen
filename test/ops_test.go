package test

import (
	"reflect"
	"testing"

	"github.com/conacts/goten/engine"
)

func TestDotProduct(t *testing.T) {
	// Create two tensors to use as input
	t1, err := engine.NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	if err != nil {
		t.Errorf("Error creating tensor: %v", err)
	}
	t2, err := engine.NewTensor([]float64{7, 8, 9, 10, 11, 12}, []int{3, 2})
	if err != nil {
		t.Errorf("Error creating tensor: %v", err)
	}

	// Call the Dot function to compute their dot product
	result, err := engine.Dot(t1, t2)
	if err != nil {
		t.Errorf("Error computing dot product: %v", err)
	}

	// Define the expected output tensor
	expected, err := engine.NewTensor([]float64{58, 64, 139, 154}, []int{2, 2})
	if err != nil {
		t.Errorf("Error creating tensor: %v", err)
	}

	// Check that the output tensor matches the expected tensor
	if !result.Equals(expected) {
		t.Errorf("Dot product test failed: expected %v, \nbut got %v\n", expected, result)
	}

	// Test for compatibility of input tensors
	t1, err = engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	if err != nil {
		t.Errorf("Error creating tensor: %v", err)
	}
	t2, err = engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	if err != nil {
		t.Errorf("Error creating tensor: %v", err)
	}
	result, err = engine.Dot(t1, t2)
	if err != nil {
		t.Errorf("Error computing dot product: %v", err)
	}
	expected, err = engine.NewTensor([]float64{7, 10, 15, 22}, []int{2, 2})
	if err != nil {
		t.Errorf("Error creating tensor: %v", err)
	}
	if !result.Equals(expected) {
		t.Errorf("Dot product test failed: expected %v, but got %v\n", expected, result)
	}
}

func TestFlatten(t *testing.T) {
	// Test case 1: flatten a 2x2 tensor
	t1, _ := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	flatT1, err := engine.Flatten(t1)
	if err != nil {
		t.Error("Flatten() returned an error:", err)
	}
	expectedData := []float64{1, 2, 3, 4}
	expectedShape := []int{4, 1}
	if !reflect.DeepEqual(flatT1.GetData(), expectedData) {
		t.Error("Flatten() returned incorrect data for a 2x2 tensor")
	}
	if !reflect.DeepEqual(flatT1.GetShape(), expectedShape) {
		t.Error("Flatten() returned incorrect shape for a 2x2 tensor")
	}

	// Test case 2: flatten a 1x5 tensor
	t2, _ := engine.NewTensor([]float64{1, 2, 3, 4, 5}, []int{1, 5})
	flatT2, err := engine.Flatten(t2)
	if err != nil {
		t.Error("Flatten() returned an error:", err)
	}
	expectedData = []float64{1, 2, 3, 4, 5}
	expectedShape = []int{5, 1}
	if !reflect.DeepEqual(flatT2.GetData(), expectedData) {
		t.Error("Flatten() returned incorrect data for a 5x1 tensor")
	}
	if !reflect.DeepEqual(flatT2.GetShape(), expectedShape) {
		t.Error("Flatten() returned incorrect shape for a 5x1 tensor")
	}

	// Test case 3: flatten a nil tensor
	_, err = engine.Flatten(nil)
	if err == nil {
		t.Error("Flatten() should have returned an error for a nil tensor")
	}
}

func TestScale(t *testing.T) {
	// Test case 1: successful element-wise multiplication
	t1, _ := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	expectedData := []float64{2, 4, 6, 8}
	expectedShape := []int{2, 2}
	expectedTensor, _ := engine.NewTensor(expectedData, expectedShape)
	result, err := engine.Scale(t1, 2)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !result.Equals(expectedTensor) {
		t.Errorf("Scale() returned incorrect result: got %v, want %v", result, expectedTensor)
	}

	// Test case 2: nil tensor
	var t2 *engine.Tensor
	result, err = engine.Scale(t2, 3)
	if err == nil {
		t.Errorf("Scale() did not return expected error for nil tensor")
	}
	if result != nil {
		t.Errorf("Scale() returned unexpected result for nil tensor: %v", result)
	}
}

func TestAdd(t *testing.T) {
	// Test case 1: add two tensors of the same shape
	t1, _ := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2, _ := engine.NewTensor([]float64{4, 3, 2, 1}, []int{2, 2})
	expected, _ := engine.NewTensor([]float64{5, 5, 5, 5}, []int{2, 2})
	result, err := engine.Add(t1, t2)
	if err != nil {
		t.Errorf("Error adding two tensors: %v", err)
	}
	if !result.Equals(expected) {
		t.Errorf("Expected %v but got %v", expected, result)
	}

	// Test case 2: add two tensors with different shapes
	t1, _ = engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2, _ = engine.NewTensor([]float64{1, 2, 3}, []int{1, 3})
	_, err = engine.Add(t1, t2)
	if err == nil {
		t.Errorf("Expected error adding tensors with different shapes but got none")
	}
}
func TestMul(t *testing.T) {
	// Test case 1: multiply two tensors of the same shape
	t1, _ := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2, _ := engine.NewTensor([]float64{4, 3, 2, 1}, []int{2, 2})
	expected, _ := engine.NewTensor([]float64{4, 6, 6, 4}, []int{2, 2})
	result, err := engine.Mul(t1, t2)
	if err != nil {
		t.Errorf("Error multiplying two tensors: %v", err)
	}
	if !result.Equals(expected) {
		t.Errorf("Expected %v but got %v", expected, result)
	}

	// Test case 2: multiply two tensors with different shapes
	t1, _ = engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	t2, _ = engine.NewTensor([]float64{1, 2, 3}, []int{1, 3})
	_, err = engine.Mul(t1, t2)
	if err == nil {
		t.Errorf("Expected error multiplying tensors with different shapes but got none")
	}
}

func TestTranspose2D(t *testing.T) {
	// Test transpose of 2x2 matrix
	t1, _ := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	expected1, _ := engine.NewTensor([]float64{1, 3, 2, 4}, []int{2, 2})
	res1, err := engine.Transpose(t1)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !res1.Equals(expected1) {
		t.Errorf("Expected %v, but got %v", expected1, res1)
	}

	/*
		// Test transpose of 3x3 matrix
		t2, _ := engine.NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{3, 3})
		expected2, _ := engine.NewTensor([]float64{1, 4, 7, 2, 5, 8, 3, 6, 9}, []int{3, 3})
		res2, err := engine.Transpose(t2)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !res2.Equals(expected2) {
			t.Errorf("Expected %v, but got %v", expected2, res2)
		}
	*/

	// Test transpose of non-square matrix
	t3, _ := engine.NewTensor([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	expected3, _ := engine.NewTensor([]float64{1, 4, 2, 5, 3, 6}, []int{3, 2})
	res3, err := engine.Transpose(t3)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !res3.Equals(expected3) {
		t.Errorf("Expected %v, but got %v", expected3, res3)
	}
}

func TestSquare(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0}
	shape := []int{2, 2}
	tensor, _ := engine.NewTensor(data, shape)

	// Test Square function
	sqTensor, err := engine.Square(tensor)
	if err != nil {
		t.Errorf("Error calling Square function: %v", err)
	}
	expectedData := []float64{1.0, 4.0, 9.0, 16.0}
	expectedShape := []int{2, 2}
	expectedTensor, _ := engine.NewTensor(expectedData, expectedShape)
	if !sqTensor.Equals(expectedTensor) {
		t.Errorf("Square function output doesn't match expected output")
	}
}

// Write tests for
// mean, min, max, neg, sub
