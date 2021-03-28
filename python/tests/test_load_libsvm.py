from time import time
from sklearn.datasets import load_svmlight_file
from napkinxc.datasets import download_dataset, load_libsvm_file
import numpy as np


def test_load_libsvm():
    datasets = {
        "eurlex-4k": {"file": "data/Eurlex/eurlex_test.txt", "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 5000, "offset": 1}},
        "amazonCat-13k": {"file": "data/AmazonCat/amazonCat_test.txt", "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 203882, "offset": 1}},
        "amazonCat-14k": {"file": "data/AmazonCat-14K/amazonCat-14K_test.txt", "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 597540, "offset": 1}},
        "wiki10-31k": {"file": "data/Wiki10/wiki10_test.txt", "sklearn_args": {"multilabel": True, "zero_based": True, "n_features": 101938, "offset": 1}}
    }

    for d, v in datasets.items():
        download_dataset(d, subset='test', format='bow')
        print("\n{} time comparison:".format(d))

        t_start = time()
        sk_X, sk_Y = load_svmlight_file(v["file"], **v["sklearn_args"])
        print("\tsklearn.datasets.load_svmlight_file time: {}s".format(time() - t_start))

        t_start = time()
        nxc_X, nxc_Y = load_libsvm_file(v["file"])
        print("\tnapkinXC.datasets.load_libsvm_file time: {}s".format(time() - t_start))

        assert np.array_equal(nxc_X.indptr, sk_X.indptr)
        assert np.array_equal(nxc_X.indices, sk_X.indices)
        assert np.allclose(nxc_X.data, sk_X.data)

        assert len(nxc_Y) == len(sk_Y)
        for nxc_y, sk_y in zip(nxc_Y, sk_Y):
            assert len(nxc_y) == len(sk_y)
            assert all(y1 == y2 for y1, y2 in zip(nxc_y, sk_y))
