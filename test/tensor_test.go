package test

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/conacts/goten/engine"
)

func TestNewTensor(t *testing.T) {
	// Test case 1: valid data and shape
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor, err := engine.NewTensor(data, shape)
	if err != nil {
		t.Errorf("Test case 1 failed: %v", err)
	}
	if !reflect.DeepEqual(tensor.GetData(), data) {
		t.Errorf("Test case 1 failed: data does not match")
	}
	if !reflect.DeepEqual(tensor.GetShape(), shape) {
		t.Errorf("Test case 1 failed: shape does not match")
	}

	// Test case 2: invalid data and shape
	data = []float64{1, 2, 3, 4, 5}
	shape = []int{2, 3}
	_, err = engine.NewTensor(data, shape)
	if err == nil {
		t.Errorf("Expected NewTensor to return an error, but it didn't")
	}
}

func TestNewRandomTensor(t *testing.T) {
	rand.Seed(42)
	shape := []int{2, 3}
	tensor, err := engine.NewRandomTensor(shape)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(tensor.GetData()) != 6 {
		t.Errorf("Expected data length to be 6, but got %d", len(tensor.GetData()))
	}
	for _, value := range tensor.GetData() {
		if value < -1 || value > 1 {
			t.Errorf("Expected value to be between -1 and 1, but got %f", value)
		}
	}
	if !reflect.DeepEqual(tensor.GetShape(), shape) {
		t.Errorf("Expected shape to be %v, but got %v", shape, tensor.GetShape())
	}
}

func TestGetShape(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor, err := engine.NewTensor(data, shape)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(tensor.GetShape(), shape) {
		t.Errorf("Expected shape %v, but got %v", shape, tensor.GetShape())
	}
}

func TestGetSize(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor, err := engine.NewTensor(data, shape)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if tensor.GetSize() != len(data) {
		t.Errorf("Expected size %d, but got %d", len(data), tensor.GetSize())
	}
}

func TestSameShape(t *testing.T) {
	data1 := []float64{1, 2, 3, 4, 5, 6}
	shape1 := []int{2, 3}
	tensor1, err := engine.NewTensor(data1, shape1)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	data2 := []float64{7, 8, 9, 10, 11, 12}
	shape2 := []int{2, 3}
	tensor2, err := engine.NewTensor(data2, shape2)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	data3 := []float64{13, 14, 15}
	shape3 := []int{3}
	tensor3, err := engine.NewTensor(data3, shape3)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	if !engine.SameShape(tensor1, tensor2) {
		t.Error("Expected SameShape(tensor1, tensor2) to return true")
	}

	if engine.SameShape(tensor1, tensor3) {
		t.Error("Expected SameShape(tensor1, tensor3) to return false")
	}
}

func TestGetData(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor, err := engine.NewTensor(data, shape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	if !reflect.DeepEqual(tensor.GetData(), data) {
		t.Errorf("Expected data %v, but got %v", data, tensor.GetData())
	}
}

func TestSetData(t *testing.T) {
	data1 := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor, err := engine.NewTensor(data1, shape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	data2 := []float64{7, 8, 9, 10, 11, 12}
	err = tensor.SetData(data2)
	if err != nil {
		t.Fatalf("Failed to set data: %v", err)
	}

	if !reflect.DeepEqual(tensor.GetData(), data2) {
		t.Errorf("Expected data %v, but got %v", data2, tensor.GetData())
	}

	data3 := []float64{12, 13, 14, 15, 16, 17, 18}
	err = tensor.SetData(data3)
	if err == nil {
		t.Errorf("Expected SetData to panic, too many elements in data to fit shape")
	}
}

func TestGetValue(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor, _ := engine.NewTensor(data, shape)

	// Test valid indices
	if val, err := tensor.GetValue([]int{0, 0}); err != nil || val != 1 {
		t.Errorf("Expected GetValue([0, 0]) to return 1, but got %v with error %v", val, err)
	}
	if val, err := tensor.GetValue([]int{0, 1}); err != nil || val != 2 {
		t.Errorf("Expected GetValue([0, 1]) to return 2, but got %v with error %v", val, err)
	}
	if val, err := tensor.GetValue([]int{1, 2}); err != nil || val != 6 {
		t.Errorf("Expected GetValue([1, 2]) to return 6, but got %v with error %v", val, err)
	}

	// Test invalid indices
	if val, err := tensor.GetValue([]int{2, 1}); err == nil {
		t.Errorf("Expected GetValue to return error with invalid indices, but got %v with value %v", err, val)
	}

	if val, err := tensor.GetValue([]int{0, -1}); err == nil {
		t.Errorf("Expected GetValue to return error with invalid indices, but got %v with value %v", err, val)
	}

	if val, err := tensor.GetValue([]int{0}); err == nil {
		t.Errorf("Expected GetValue to return error with invalid indices, but got %v with value %v", err, val)
	}

	if val, err := tensor.GetValue([]int{0, 1, 2}); err == nil {
		t.Errorf("Expected GetValue to return error with invalid indices, but got %v with value %v", err, val)
	}
}

func TestSetValue(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	shape := []int{2, 3}
	tensor, _ := engine.NewTensor(data, shape)

	// Test valid indices
	err := tensor.SetValue(10, []int{0, 0})
	if err != nil {
		t.Errorf("Expected SetValue(10, [0, 0]) to succeed, but got error: %v", err)
	}
	if value, _ := tensor.GetValue([]int{0, 0}); value != 10 {
		t.Errorf("Expected SetValue(10, [0, 0]) to set the value to 10, but got %v", value)
	}

	err = tensor.SetValue(20, []int{1, 2})
	if err != nil {
		t.Errorf("Expected SetValue(20, [1, 2]) to succeed, but got error: %v", err)
	}
	if value, _ := tensor.GetValue([]int{1, 2}); value != 20 {
		t.Errorf("Expected SetValue(20, [1, 2]) to set the value to 20, but got %v", value)
	}

	// Test invalid indices
	err = tensor.SetValue(10, []int{2, 1})
	if err == nil {
		t.Error("Expected SetValue to panic with invalid indices, but it didn't")
	}

	err = tensor.SetValue(10, []int{0, -1})
	if err == nil {
		t.Error("Expected SetValue to panic with invalid indices, but it didn't")
	}

	err = tensor.SetValue(10, []int{0})
	if err == nil {
		t.Error("Expected SetValue to panic with invalid indices, but it didn't")
	}

	err = tensor.SetValue(10, []int{0, 1, 2})
	if err == nil {
		t.Error("Expected SetValue to panic with invalid indices, but it didn't")
	}
}

func TestEncode(t *testing.T) {
	// Test case 1: valid inputs
	coords := []int{1, 2}
	shape := []int{4, 3}
	expectedIndex := 5
	index, err := engine.EncodePos(coords, shape)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if index != expectedIndex {
		t.Errorf("Encode() returned unexpected index: got %d, expected %d", index, expectedIndex)
	}

	// Test case 2: invalid inputs
	coords = []int{1, 5}
	shape = []int{4, 3}
	_, err = engine.EncodePos(coords, shape)
	if err == nil {
		t.Error("Encode() did not return an error for out-of-range coordinates")
	}
}

func TestDecode(t *testing.T) {
	// Test case 1: valid input
	index := 5
	shape := []int{4, 3}
	expectedCoords := []int{1, 2}
	coords, err := engine.DecodePos(index, shape)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(coords, expectedCoords) {
		t.Errorf("Decode() returned unexpected coordinates: got %v, expected %v", coords, expectedCoords)
	}

	// Test case 2: invalid input
	index = 14
	shape = []int{4, 3}
	_, err = engine.DecodePos(index, shape)
	if err == nil {
		t.Error("Decode() did not return an error for out-of-range index")
	}
}

func TestEquals(t *testing.T) {
	// Test case 1: equal tensors
	t1, err1 := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	if err1 != nil {
		t.Errorf("Error creating tensor: %v", err1)
	}
	t2, err2 := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	if err2 != nil {
		t.Errorf("Error creating tensor: %v", err2)
	}
	if !t1.Equals(t2) {
		t.Error("Equals() returned false for equal tensors")
	}

	// Test case 2: unequal data
	t5, err5 := engine.NewTensor([]float64{1, 2, 3, 4}, []int{2, 2})
	if err5 != nil {
		t.Errorf("Error creating tensor: %v", err5)
	}
	t6, err6 := engine.NewTensor([]float64{1, 2, 5, 4}, []int{2, 2})
	if err6 != nil {
		t.Errorf("Error creating tensor: %v", err6)
	}
	if t5.Equals(t6) {
		t.Error("Equals() returned true for tensors with unequal data")
	}
}
