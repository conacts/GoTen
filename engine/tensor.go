package engine

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
)

// COULD CHANGE `data` TO HOLD ACTUAL ARRAYS
type Tensor struct {
	data  []float64 // The data stored in the tensor
	shape []int     // The shape of the tensor, [row, col, ...]
	dw    *Tensor   // Gradients of the weights
	db    *Tensor   // Gradients of the biases
	dx    *Tensor   // Gradient of the tensor
}

func NewTensor(data []float64, shape []int) (*Tensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
		if dim == 0 {
			return nil, fmt.Errorf("invalid shape: shape contains a zero %v", shape)
		}
	}
	if size != len(data) {
		return nil, fmt.Errorf("data size %d does not match shape %v", len(data), shape)
	}
	t := &Tensor{
		data:  data,
		shape: shape,
		dw:    nil,
		db:    nil,
		dx:    nil,
	}
	return t, nil
}

func NewRandomTensor(shape []int) (*Tensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
		if dim == 0 {
			return nil, fmt.Errorf("invalid shape: shape contains a zero %v", shape)
		}
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.Float64()*2 - 1
	}
	return NewTensor(data, shape)
}

func NewZeroTensor(shape []int) (*Tensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
		if dim == 0 {
			return nil, fmt.Errorf("invalid shape: shape contains a zero %v", shape)
		}
	}
	data := make([]float64, size)
	return NewTensor(data, shape)
}

// Returns the shape of the tensor as a slice of integers.
func (t *Tensor) GetShape() []int {
	if t.shape == nil {
		return nil
	}
	return t.shape
}

// Returns true if t and other have the same shape, false otherwise.
func SameShape(t1, t2 *Tensor) bool {
	if t1.shape == nil || t2.shape == nil {
		return false
	}

	if len(t1.shape) != len(t2.shape) {
		return false
	}

	for i := range t1.shape {
		if t1.shape[i] != t2.shape[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) Reshape(shape []int) error {
	size := 1
	for _, dim := range shape {
		size *= dim
		if dim == 0 {
			return fmt.Errorf("invalid shape: shape contains a zero %v", shape)
		}
	}
	if size != t.GetSize() {
		return fmt.Errorf("invalid shape: new shape %v is not of size %d", shape, t.GetSize())
	}
	t.shape = shape
	return nil
}

// Returns the total number of elements in the tensor.
func (t *Tensor) GetSize() int {
	if t.shape == nil {
		return 0
	}

	size := 1
	for _, dim := range t.shape {
		if dim <= 0 {
			return 0
		}
		size *= dim
	}
	return size
}

// Returns a copy of the data stored in the tensor
func (t *Tensor) GetData() []float64 {
	if t.data == nil {
		return nil
	}
	return t.data
}

func (t *Tensor) SetData(data []float64) error {
	if t.shape == nil {
		return fmt.Errorf("tensor shape is nil")
	}
	size := t.GetSize()
	if len(data) != size {
		return fmt.Errorf("data size does not match tensor size: tensor: %d, other: %d", size, len(data))
	}
	t.data = data
	return nil
}

func (t *Tensor) GetValue(indices []int) (float64, error) {
	if t.shape == nil {
		return 0, fmt.Errorf("tensor shape is nil")
	}
	if len(indices) != len(t.shape) {
		return 0, fmt.Errorf("invalid index length for tensor shape")
	}

	index := 0
	for i, dim := range t.shape {
		if indices[i] >= dim || indices[i] < 0 {
			return 0, fmt.Errorf("index out of range")
		}
		index = index*dim + indices[i]
	}
	return t.data[index], nil
}

func (t *Tensor) SetValue(value float64, index []int) error {
	if len(index) != len(t.shape) {
		return fmt.Errorf("invalid number of indices: expected %d, got %d", len(t.shape), len(index))
	}
	shape := t.GetShape()
	encodedIndex, err2 := EncodePos(index, shape)
	if err2 != nil {
		return fmt.Errorf("error encoding index: %v", err2)
	}
	t.data[encodedIndex] = value
	return nil
}

func EncodePos(coords []int, shape []int) (int, error) {
	if len(shape) != len(coords) {
		return 0, fmt.Errorf("shape and coords have different lengths: %v vs %v", shape, coords)
	}
	index := 0
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if coords[i] < 0 || coords[i] >= shape[i] {
			return 0, fmt.Errorf("index out of range for dimension %d: expected 0 to %d, got %d", i, shape[i]-1, coords[i])
		}
		index += coords[i] * stride
		stride *= shape[i]
	}
	return index, nil
}

func DecodePos(index int, shape []int) ([]int, error) {
	sum := 1
	for _, s := range shape {
		sum *= s
	}
	if index < 0 || index >= sum {
		return nil, fmt.Errorf("index out of range: expected 0 to %d, got %d", sum-1, index)
	}
	coords := make([]int, len(shape))
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	for i := 0; i < len(shape); i++ {
		coords[i] = (index / strides[i]) % shape[i]
	}
	return coords, nil
}

func (t1 *Tensor) Equals(t2 *Tensor) bool {
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		return false
	}
	if !reflect.DeepEqual(t1.data, t2.data) {
		return false
	}
	return true
}

func (t *Tensor) GetWeightGrads() *Tensor {
	return t.dw
}

func (t *Tensor) SetWeightGrads(dw *Tensor) {
	t.dw = dw
}

func (t *Tensor) GetBiasesGrads() *Tensor {
	return t.db
}

func (t *Tensor) SetBiasGrads(db *Tensor) {
	t.db = db
}

func (t *Tensor) GetInputGrads() *Tensor {
	return t.dx
}

func (t *Tensor) SetInputGrads(dx *Tensor) {
	t.dx = dx
}

// String returns a string representation of the tensor.
// Currently changes the shape due to logic error in `StringHelper`
func (t *Tensor) String() string {
	shape := append([]int{t.shape[1], t.shape[0]}, t.shape[2:]...)
	return "[" + t.StringHelper(t.data, shape, len(shape)-1, 0) + "]"
}

// Recursively builds string of tensor data
func (t *Tensor) StringHelper(data []float64, shape []int, dim, index int) string {
	var sb strings.Builder
	if dim == 0 {
		for i, v := range data {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%.4f", v))
		}
	} else {
		for i := 0; i < shape[dim]; i++ {
			if i > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString("[")
			sb.WriteString(t.StringHelper(data[index:index+shape[dim-1]], shape, dim-1, index))
			sb.WriteString("]")
			index += shape[dim-1]
		}
	}
	return sb.String()
}

func IsNumIn(n int, list []int) bool {
	for _, num := range list {
		if num == n {
			return true
		}
	}
	return false
}
