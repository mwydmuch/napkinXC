#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


def remap_labels(y, mapping):
    if type(y) is list:
        for i in range(len(y)):
            new_y = []
            for yij in y[i]:
                new_id = mapping.get(yij, len(mapping))
                if new_id == len(mapping):
                    mapping[yij] = new_id
                new_y.append(new_id)
            y[i] = tuple(new_y)
    else:
        for i in range(y.shape[0]):
            new_id = mapping.get(y[i], len(mapping))
            if new_id == len(mapping):
                mapping[y[i]] = new_id
            y[i] = new_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: remap_libsvm.py [file ...]")
        exit(1)

    mapping = {}
    for file in sys.argv[1:]:
        X, y = load_svmlight_file(file, multilabel=True)
        remap_labels(y, mapping)
        dump_svmlight_file(X, y, file + ".remapped", multilabel=True)

    print("Max label id in original mapping:", max(mapping, key=int))
    print("Max label id after remapping:", len(mapping))
