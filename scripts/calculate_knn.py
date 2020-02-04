#!/usr/bin/env python3

import sys
from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics.pairwise import cosine_similarity


def tmp_dump_svmlight_file (X, y, output):
    cX = coo_matrix(X)
    i = 0
    prev_row = 0
    with open(output, "w") as file:
        for row_y in y:
            file.write(','.join([str(l) for l in row_y]))
            while i < cX.row.shape[0] and prev_row == cX.row[i]:
                file.write(" {}:{}".format(cX.col[i], cX.data[i]))
                i += 1
            file.write("\n")
            if i < cX.row.shape[0]:
                prev_row = cX.row[i]

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: calculate_knn.py [input train] [input test] [output_test] [k]")
        exit(1)

    input_train = sys.argv[1]
    input_test = sys.argv[2]
    output_test = sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    train_X, train_y = load_svmlight_file(input_train, multilabel=True)
    test_X, test_y = load_svmlight_file(input_test, multilabel=True)

    if train_X.shape[1] != test_X.shape[1]:
        test_X.resize((test_X.shape[0], train_X.shape[1]))

    cos_sim = cosine_similarity(test_X, train_X)
    test_y = []

    for row in cos_sim:
        test_y.append(list(row.argsort()[-k:]))

    tmp_dump_svmlight_file(test_X, test_y, output_test)
    #dump_svmlight_file(test_X, test_y, output_test, multilabel=True)
