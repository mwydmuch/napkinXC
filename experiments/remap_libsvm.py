#!/usr/bin/env python3

import sys
#import numpy as np
#from sklearn.datasets import load_svmlight_file, dump_svmlight_file


def load_libsvm(file):
    X = []
    Y = []
    with open(file) as f:
        for row in f:
            y, x = row.split(' ', 1)
            X.append(x)
            if len(y):
                Y.append(y.split(','))
            else:
                Y.append([])

    assert len(X) == len(Y)
    return X, Y


def save_libsvm(X, Y, file):
    with open(file, "w") as f:
        for x, y in zip(X, Y):
            f.write(','.join([str(y_i) for y_i in sorted(y)]))
            f.write(' ')
            f.write(x)


def remap_files(files, mapping):
    for file in files:
        X, Y = load_libsvm(file)
        remap_labels(Y, mapping)
        save_libsvm(X, Y, file + ".remapped")


# def remap_files_sklearn(files, mapping):
#     for file in files:
#         X, Y = load_svmlight_file(file, multilabel=True)
#         remap_labels(Y, mapping)
#         dump_svmlight_file(X, Y, file + ".remapped", multilabel=True)


def remap_labels(y, mapping):
    if type(y) is list:
        for i in range(len(y)):
            new_y = []
            for yij in y[i]:
                new_id = mapping.get(yij, len(mapping))
                if new_id == len(mapping):
                    mapping[yij] = new_id
                new_y.append(new_id)
            y[i] = new_y #tuple(new_y)
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
    #remap_files_sklearn(sys.argv[1:], mapping)
    remap_files(sys.argv[1:], mapping)

    print("Max label id in original mapping:", max(mapping, key=int))
    print("Max label id after remapping:", len(mapping))
