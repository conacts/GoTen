package engine

import (
	"errors"
	"fmt"
	"math"
	"reflect"
)

// ----------------------- TENSOR -----------------------

func Flatten(t *Tensor) (*Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("cannot flatten nil tensor")
	}
	data := t.GetData()
	if data == nil {
		return nil, fmt.Errorf("cannot flatten tensor with nil data")
	}
	shape := t.GetShape()
	if shape == nil {
		return nil, fmt.Errorf("cannot flatten tensor with nil shape")
	}
	return NewTensor(data, []int{len(data), 1})
}

func Scale(t *Tensor, v float64) (*Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("cannot perform element-wise multiplication on nil tensor")
	}
	data := t.GetData()
	if data == nil {
		return nil, fmt.Errorf("error getting tensor data")
	}
	shape := t.GetShape()
	newData := make([]float64, len(data))
	for i, val := range data {
		newData[i] = val * v
	}
	return NewTensor(newData, shape)
}

// Add returns a new tensor that is the elementwise sum of this tensor and another tensor.
// The tensors must have the same shape.
func Add(t1, t2 *Tensor) (*Tensor, error) {
	if t1 == nil || t2 == nil {
		return nil, fmt.Errorf("cannot perform element-wise addition with nil tensor")
	}
	if !SameShape(t1, t2) {
		return nil, fmt.Errorf("cannot add tensors with different shapes { t1: %v and t2: %v }", t1.GetShape(), t2.GetShape())
	}
	data1 := t1.GetData()
	if data1 == nil {
		return nil, fmt.Errorf("error getting data from tensor 1")
	}
	data2 := t2.GetData()
	if data2 == nil {
		return nil, fmt.Errorf("error getting data from tensor 2")
	}
	data := make([]float64, t1.GetSize())
	for i := 0; i < t1.GetSize(); i++ {
		data[i] = data1[i] + data2[i]
	}

	return NewTensor(data, t1.GetShape())
}

// Returns a new tensor that is the element-wise product of t1 and t2.
func Mul(t1, t2 *Tensor) (*Tensor, error) {
	if t1 == nil || t2 == nil {
		return nil, fmt.Errorf("cannot perform element-wise multiplication with nil tensor")
	}
	if !SameShape(t1, t2) {
		return nil, fmt.Errorf("tensors must have the same shape to perform element-wise multiplication: %v and %v", t1.GetShape(), t2.GetShape())
	}
	data := make([]float64, len(t1.data))
	for i := range t1.data {
		data[i] = t1.data[i] * t2.data[i]
	}
	return NewTensor(data, t1.shape)
}

// Dot returns a new tensor that is the matrix product of t1 and t2.
// This is not a dot product...
func Dot(t1, t2 *Tensor) (*Tensor, error) {
	// Check that the input tensors have compatible shapes
	if len(t1.GetShape()) != 2 || len(t2.GetShape()) != 2 || t1.GetShape()[1] != t2.GetShape()[0] {
		return nil, fmt.Errorf("incompatible shapes for dot product: t1: %v and t2: %v", t1.GetShape(), t2.GetShape())
	}

	// Create a new tensor to hold the result
	shape := []int{t1.GetShape()[0], t2.GetShape()[1]}
	result, err := NewZeroTensor(shape)
	if err != nil {
		return nil, err
	}

	// Compute the dot product for each pair of rows and columns
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			sum := 0.0
			for k := 0; k < t1.GetShape()[1]; k++ {
				v1, err1 := t1.GetValue([]int{i, k})
				if err1 != nil {
					return nil, fmt.Errorf("error getting value from tensor 1: %v", err1)
				}
				v2, err2 := t2.GetValue([]int{k, j})
				if err2 != nil {
					return nil, fmt.Errorf("error getting value from tensor 2: %v", err2)
				}
				sum += v1 * v2
			}
			result.SetValue(sum, []int{i, j})
		}
	}
	return result, nil
}

// Transpose returns a new tensor that is the transpose of the input tensor.
func Transpose(t *Tensor) (*Tensor, error) {
	shape := t.GetShape()
	data := t.GetData()
	if len(shape) == 2 {
		rows, cols := shape[0], shape[1]
		transposed := make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				transposed[j*rows+i] = data[i*cols+j]
			}
		}
		return NewTensor(transposed, []int{shape[1], shape[0]})
	} else if len(shape) == 3 {
		return nil, fmt.Errorf("transpose is not defined for tensors with shape %v", shape)
	} else {
		return nil, fmt.Errorf("transpose is not defined for tensors with shape %v", shape)
	}
}

// Relu applies the rectified linear unit (ReLU) function element-wise to the tensor.
func Relu(t *Tensor) (*Tensor, error) {
	data := make([]float64, len(t.data))
	for i, v := range t.data {
		if v > 0 {
			data[i] = v
		} else {
			data[i] = 0
		}
	}
	return NewTensor(data, t.shape)
}

func Sigmoid(t *Tensor) (*Tensor, error) {
	data := make([]float64, len(t.data))
	for i, v := range t.data {
		data[i] = 1 / (1 + math.Exp(-v))
	}
	return NewTensor(data, t.shape)
}

// Square computes the element-wise square of the input tensor.
func Square(t *Tensor) (*Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("input tensor is nil")
	}
	data := t.GetData()
	sqData := make([]float64, len(data))
	for i := 0; i < len(data); i++ {
		sqData[i] = data[i] * data[i]
	}
	out, _ := NewTensor(sqData, t.GetShape())
	return out, nil
}

// Mean computes the mean of each row of the input tensor and returns a tensor of shape (1, lenout).
func Mean(t *Tensor) (*Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("input tensor is nil")
	}

	shape := t.GetShape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input tensor must have 2 dimensions")
	}
	rows, cols := shape[0], shape[1]

	meanData := make([]float64, rows)
	for i := 0; i < rows; i++ {
		rowStart := i * cols
		rowEnd := rowStart + cols
		rowSum := 0.0
		for j := rowStart; j < rowEnd; j++ {
			rowSum += t.GetData()[j]
		}
		meanData[i] = rowSum / float64(cols)
	}

	meanTensor, _ := NewTensor(meanData, []int{rows, 1})
	return meanTensor, nil
}

func Max(t *Tensor) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("input tensor is nil")
	}
	data := t.GetData()
	if len(data) == 0 {
		return nil, fmt.Errorf("input tensor has no elements")
	}
	shape := t.GetShape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input tensor should have 2 dimensions")
	}
	maxData := make([]float64, shape[0])
	for i := 0; i < shape[0]; i++ {
		max := math.Inf(-1)
		for j := 0; j < shape[1]; j++ {
			if data[i*shape[1]+j] > max {
				max = data[i*shape[1]+j]
			}
		}
		maxData[i] = max
	}
	out, _ := NewTensor(maxData, []int{shape[0], 1})
	return out, nil
}

func Min(t *Tensor) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("input tensor is nil")
	}
	data := t.GetData()
	if len(data) == 0 {
		return nil, fmt.Errorf("input tensor has no elements")
	}
	shape := t.GetShape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input tensor should have 2 dimensions")
	}
	minData := make([]float64, shape[0])
	for i := 0; i < shape[0]; i++ {
		min := math.Inf(1)
		for j := 0; j < shape[1]; j++ {
			if data[i*shape[1]+j] < min {
				min = data[i*shape[1]+j]
			}
		}
		minData[i] = min
	}
	out, _ := NewTensor(minData, []int{shape[0], 1})
	return out, nil
}

func Neg(t *Tensor) (*Tensor, error) {
	data := t.GetData()
	outdata := make([]float64, len(data))
	for i := 0; i < len(data); i++ {
		outdata[i] = -data[i]
	}
	out, err := NewTensor(outdata, t.GetShape())
	if err != nil {
		return nil, fmt.Errorf("failed to create negated tensor in Neg(): %v", err)
	}
	return out, nil
}

// Sub subtracts tensor y from tensor x.
func Sub(t1, t2 *Tensor) (*Tensor, error) {
	if !reflect.DeepEqual(t1.shape, t2.shape) {
		return nil, fmt.Errorf("tensors are not of the same shape t1: %v compare to t2: %v", t1.GetShape(), t2.GetShape())
	}

	// Negate y and add it to x
	negt2, err := Neg(t2)
	if err != nil {
		return nil, fmt.Errorf("failed to negate tensor t2: %v", err)
	}
	result, err := Add(t1, negt2)
	if err != nil {
		return nil, fmt.Errorf("failed to add tensors: %v", err)
	}

	return result, nil
}

// Exp applies the exponential function element-wise to the input tensor.
func Exp(t *Tensor) (*Tensor, error) {
	// Create a new tensor to hold the output values
	out, err := NewZeroTensor(t.GetShape())
	if err != nil {
		return nil, err
	}

	// Apply the exponential function element-wise to the input tensor
	for i, x := range t.GetData() {
		out.data[i] = math.Exp(x)
	}

	return out, nil
}

func SumCols(t *Tensor) (*Tensor, error) {
	shape := t.GetShape()
	data := t.GetData()
	if len(shape) < 2 {
		return nil, fmt.Errorf("cannot sum columns of a tensor with less than 2 dimensions")
	}
	cols := shape[len(shape)-1]
	rows := 1
	if len(shape) > 2 {
		rows = len(shape[:len(shape)-1])
	}
	colSums := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			colSums[j] += data[i*cols+j]
		}
	}
	out, err := NewTensor(colSums, []int{1, cols})
	if err != nil {
		return nil, err
	}
	return out, nil
}
