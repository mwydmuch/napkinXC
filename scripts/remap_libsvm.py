#!/usr/bin/env python3

import sys
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


def remap_labels(y, mapping):
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
        dump_svmlight_file(X, y, file + ".remapped")

    print("Max id of label in original mapping:", max(mapping, key=int))
    print("Max id of label in after remapping:", len(mapping))
