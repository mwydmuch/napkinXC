#!/usr/bin/env python3

import sys
from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: evaluate_knn.py [input train] [input test] [input prediction]")
        exit(1)

    input_train = sys.argv[1]
    input_test = sys.argv[2]
    input_prediction = sys.argv[3]

    train_X, train_y = load_svmlight_file(input_train, multilabel=True)
    test_X, test_y = load_svmlight_file(input_test, multilabel=True)

    with open(input_prediction) as prediction_file:
        sim = 0.0
        for i, pred_row in tqdm(enumerate(prediction_file)):
            pred = int(pred_row.strip().split(" ")[0])
            sim += cosine_similarity(train_X[pred - 1], test_X[i])[0][0]

        print("Mean similarity: {}".format(sim / i))
