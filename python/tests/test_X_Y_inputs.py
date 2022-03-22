import shutil
from time import time
import pytest
import numpy as np
from scipy.sparse import csr_matrix

from napkinxc.datasets import load_dataset, to_csr_matrix, to_np_matrix
from napkinxc.models import HSM, PLT
from napkinxc.measures import precision_at_k

from conf import *
MODEL_PATH = get_model_path(__file__)


def print_data_info(X):
    if not isinstance(X, list):
        print(f"Type (shape, dtype): {type(X)} ({X.shape}, {X.dtype})")
        if isinstance(X, csr_matrix):
            print(f"Internal types: indptr={type(X.indptr.dtype)}, indices={type(X.indices.dtype)}, data={type(X.data.dtype)}")
    else:
        print(f"Type (len): {type(X)} ({len(X)}), {X[0]}, X[0] type: {type(X[0])})")  #, X[0][0] type: {type(X[0][0])}")


def eval_data_for_dataset(X_train, Y_train, model_class):
    print_data_info(X_train)
    print_data_info(Y_train)
        
    model = model_class(MODEL_PATH, optimizer="adagrad", epochs=1, tree_type="huffman")

    t_start = time()
    model.fit(X_train, Y_train)
    print(f"Train time: {time() - t_start}")

    X_test, Y_test = load_dataset(TEST_DATASET, "test", root=TEST_DATA_PATH)

    t_start = time()
    Y_pred = model.predict(X_test, top_k=1)
    print(f"Predict time: {time() - t_start}")

    p_at_1 = precision_at_k(Y_test, Y_pred, k=1)
    print(f"P@1: {p_at_1}")
    assert 0.25 < p_at_1 < SCORE_RANGE[1]

    shutil.rmtree(MODEL_PATH, ignore_errors=True)


def test_multilabel_list_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    eval_data_for_dataset(X_train, Y_train, PLT)


def test_multiclass_list_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    X_train = X_train
    Y_train = [y[0] if len(y) else 0 for y in Y_train]
    eval_data_for_dataset(X_train, Y_train, HSM)


def test_multilabel_numpy_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    #X_train = X_train.toarray()
    X_train = to_np_matrix(X_train)
    Y_train = to_np_matrix(Y_train)
    eval_data_for_dataset(X_train, Y_train, PLT)


def test_multilabel_numpy_float64_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    #X_train = X_train.toarray()
    X_train = to_np_matrix(X_train, dtype=np.float64)
    Y_train = to_np_matrix(Y_train, dtype=np.float64)
    eval_data_for_dataset(X_train, Y_train, PLT)


def test_multiclass_numpy_float64_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    #X_train = X_train.toarray()
    X_train = to_np_matrix(X_train, dtype=np.float64)
    Y_train = np.array([y[0] if len(y) else 0 for y in Y_train], dtype=np.float64)
    eval_data_for_dataset(X_train, Y_train, HSM)


def test_multiclass_numpy_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    X_train = to_np_matrix(X_train, dtype=np.float64)
    Y_train = np.array([y[0] if len(y) else 0 for y in Y_train], dtype=np.int32)
    eval_data_for_dataset(X_train, Y_train, HSM)


def test_multiclass_numpy_int64_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    X_train = to_np_matrix(X_train, dtype=np.float64)
    Y_train = np.array([y[0] if len(y) else 0 for y in Y_train], dtype=np.int64)
    eval_data_for_dataset(X_train, Y_train, HSM)



def test_multilabel_csr_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    Y_train = to_csr_matrix(Y_train)
    eval_data_for_dataset(X_train, Y_train, PLT)


def test_multilabel_csr_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    Y_train = to_csr_matrix(Y_train)
    eval_data_for_dataset(X_train, Y_train, PLT)


def test_multilabel_csr_float64_input():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)

    X_train = to_csr_matrix(X_train, dtype=np.float64)
    Y_train = to_csr_matrix(Y_train)
    eval_data_for_dataset(X_train, Y_train, PLT)


def test_numpy_3d_input():
    size = 100
    X_train = np.ones((size, size, size))
    Y_train = np.ones((size))

    print(f"Type (shape, dtype): {type(X_train)} ({X_train.shape}, {X_train.dtype})")
    print(f"Type (shape, dtype): {type(Y_train)} ({Y_train.shape}, {Y_train.dtype})")

    with pytest.raises(ValueError):
        model = PLT(MODEL_PATH, optimizer="adagrad", epochs=1)
        model.fit(X_train, Y_train)
