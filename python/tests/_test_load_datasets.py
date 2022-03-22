from time import time
import os
from napkinxc.datasets import *


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-data")


def test_load_dataset():
    for d in DATASETS.values():
        for f in d['formats']:
            for s in d['subsets']:
                download_dataset(d['name'], subset=s, format=f, root=data_path)
                t_start = time()
                X, Y = load_dataset(d['name'], subset=s, format=f, root=data_path)
                len_X = len(X) if isinstance(X, list) else X.shape[0]
                len_Y = len(Y) if isinstance(Y, list) else Y.shape[0]
                assert len_X == len_Y
                print("\tload_dataset({}, subset={}, format={}) rows: {}, time: {}s\n".format(d['name'], s, f, len_X, time() - t_start))
