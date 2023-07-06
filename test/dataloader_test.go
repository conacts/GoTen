package test

import (
	"reflect"
	"testing"

	"github.com/conacts/goten/dataloader"
)

func TestLoadData(t *testing.T) {
	// Test with a valid file path
	path := "../testdata/sample.csv"
	expected := [][]string{
		{"1", "2", "3"},
		{"4", "5", "6"},
		{"7", "8", "9"},
	}
	got, err := dataloader.LoadData(path)
	if err != nil {
		t.Fatalf("Unexpected error while loading data: %v", err)
	}
	if !reflect.DeepEqual(expected, got) {
		t.Fatalf("Unexpected output from LoadData.\nExpected: %v\nGot: %v", expected, got)
	}

	// Test with an invalid file path
	path = "../testdata/nonexistent.csv"
	expectedErr := "unable to read input file ../testdata/nonexistent.csv: open ../testdata/nonexistent.csv: no such file or directory"
	_, err = dataloader.LoadData(path)
	if err == nil || err.Error() != expectedErr {
		t.Fatalf("Unexpected output from LoadData.\nExpected error: %v\nGot error: %v", expectedErr, err)
	}

	// Test with an invalid CSV file
	/*
		path = "../testdata/invalid.csv"
		expectedErr = "Unable to parse file as CSV for ../testdata/invalid.csv"
		_, err = dataloader.LoadData(path)
		if err == nil || err.Error() != expectedErr {
			t.Fatalf("Unexpected output from LoadData.\nExpected error: %v\nGot error: %v", expectedErr, err)
		}
	*/
}
