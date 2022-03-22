from time import time
import os
from sklearn.datasets import load_svmlight_file
from napkinxc.datasets import download_dataset, load_libsvm_file
import numpy as np

from conf import *
MODEL_PATH = get_model_path(__file__)


def test_load_libsvm():
    datasets = {
        "eurlex-4k": {"file": os.path.join(TEST_DATA_PATH, "Eurlex/eurlex_test.txt"), "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 5000, "offset": 1}},
        "amazonCat-13k": {"file": os.path.join(TEST_DATA_PATH, "AmazonCat/amazonCat_test.txt"), "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 203882, "offset": 1}},
        "amazonCat-14k": {"file": os.path.join(TEST_DATA_PATH, "AmazonCat-14K/amazonCat-14K_test.txt"), "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 597540, "offset": 1}},
        "wiki10-31k": {"file": os.path.join(TEST_DATA_PATH, "Wiki10/wiki10_test.txt"), "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 101938, "offset": 1}}
    }

    for d, v in datasets.items():
        download_dataset(d, subset='test', format='bow', root=TEST_DATA_PATH)
        print("\n{} time comparison:".format(d))

        t_start = time()
        sk_X, sk_Y = load_svmlight_file(v["file"], **v["sklearn_args"])
        print("\tsklearn.datasets.load_svmlight_file time: {}s".format(time() - t_start))

        t_start = time()
        nxc_X1, nxc_Y_list = load_libsvm_file(v["file"], labels_format="list")
        print("\tnapkinXC.datasets.load_libsvm_file time: {}s".format(time() - t_start))

        t_start = time()
        nxc_X2, nxc_Y_csrm = load_libsvm_file(v["file"], labels_format="csr_matrix")
        print("\tnapkinXC.datasets.load_libsvm_file time: {}s".format(time() - t_start))

        assert np.array_equal(nxc_X1.indptr, nxc_X2.indptr)
        assert np.array_equal(nxc_X1.indices, nxc_X2.indices)
        assert np.array_equal(nxc_X1.data, nxc_X2.data)

        assert np.array_equal(nxc_X1.indptr, sk_X.indptr)
        assert np.array_equal(nxc_X1.indices, sk_X.indices)
        assert np.allclose(nxc_X1.data, sk_X.data)
        assert nxc_X1.shape[0] == nxc_Y_csrm.shape[0]

        assert len(nxc_Y_list) == len(sk_Y)
        for nxc_y, sk_y in zip(nxc_Y_list, sk_Y):
            assert len(nxc_y) == len(sk_y)
            assert all(y1 == y2 for y1, y2 in zip(nxc_y, sk_y))
