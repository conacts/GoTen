package dataloader

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/conacts/goten/engine"
)

// LoadData loads data from csv files at the specified file paths.
// It returns two slices, Xout and yout, where Xout contains the features as strings,
// and yout contains the target values as strings.
func LoadData(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("unable to read input file %s: %v", path, err)
	}
	defer f.Close()

	fout, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("unable to parse file as CSV for %s: %v", path, err)
	}

	return fout, nil
}

func EncodeCSVToTensorList(csv [][]string) ([]*engine.Tensor, error) {
	rows, cols := len(csv), len(csv[0])
	out := make([]*engine.Tensor, rows)
	for i := 0; i < len(csv); i++ {
		data := make([]float64, cols)
		for j, cell := range csv[i] {
			if num, err := strconv.ParseFloat(cell, 64); err == nil {
				data[j] = num
			} else {
				return nil, fmt.Errorf("error in %s at row: %d and col: %d", cell, i, j)
			}
		}
		var err error
		out[i], err = engine.NewTensor(data, []int{1, cols})
		if err != nil {
			return nil, fmt.Errorf("error creating tensor: %v", err)
		}
	}
	return out, nil
}

/*
func EncodeCSVToTensorList(csv [][]string, batchSize int) ([][]*engine.Tensor, error) {
	rows, cols := len(csv), len(csv[0])
	numBatches := int(math.Ceil(float64(rows) / float64(batchSize)))
	out := make([][]*engine.Tensor, numBatches)

	for i := 0; i < numBatches; i++ {
		start := i * batchSize
		end := int(math.Min(float64(start+batchSize), float64(rows)))
		batchSize := end - start
		batch := make([]*engine.Tensor, batchSize)
		for j := 0; j < batchSize; j++ {
			row := start + j
			data := make([]float64, cols)
			for k, cell := range csv[row] {
				if num, err := strconv.ParseFloat(cell, 64); err == nil {
					data[k] = num
				} else {
					return nil, fmt.Errorf("error in %s at row: %d and col: %d", cell, row, k)
				}
			}
			var err error
			batch[j], err = engine.NewTensor(data, []int{1, cols})
			if err != nil {
				return nil, fmt.Errorf("error creating tensor: %v", err)
			}
		}
		out[i] = batch
	}
	return out, nil
}
*/
