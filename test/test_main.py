import pytest
from scripts.main import *


def test_get_data():
    data = get_data()
    train_df = data[0]
    test_df = data[1]
    assert train_df.shape[1] == 12, "Error in train dataframe shape (number of columns)"
    assert test_df.shape[1] == 11, "Error in test dataframe shape (number of columns)"

def test_split_data():
    data = get_data()[0]
    X, y = split_data(data)
    assert X.shape[1] == 11, "Error in features shape (number of columns)"
    assert y.shape[0] == X.shape[0], "Error in labels size"

def test_process_data():
    data = get_data()
    x, y = split_data(data[0])
    x_processed = process_data(x)[0]
    assert x_processed.shape[0] == y.shape[0], "Error in shape for process_data, the shape doesnt fit with shape for the labels"

def test_get_predictions():
    train_df, test_df = get_data()
    X, y = split_data(train_df)
    x_processed = process_data(X)[0]
    best_model = get_best_model(x_processed, y)
    x_test = process_data(test_df)[0]
    predictions = get_predictions(best_model, x_test)
    assert x_test.shape[0] == predictions.shape[0] , "error, the shape between predictions and x_test does not match"   