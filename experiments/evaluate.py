#!/usr/bin/env python

import sys
import os

napkinxc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../python")
sys.path.append(napkinxc_path)

from napkinxc.measures import *


def load_true_file(filepath):
    with open(filepath) as file:
        Y = []
        for i, line in enumerate(file):
            if i == 0 and len(line.split(' ')) == 3:
                continue
            Y.append([int(y) for y in line.strip().split(' ', 1)[0].split(',') if ':' not in y])
        return Y


def load_pred_file(filepath):
    with open(filepath) as file:
        Y = []

        def convert_y(y):
            y = y.split(':')
            if len(y) == 2:
                return (int(y[0]), float(y[1]))
            else:
                return int(y)

        for line in file:
            Y.append([convert_y(y) for y in line.strip().split(' ')])
        return Y


def load_inv_ps_file(filepath):
    with open(filepath) as file:
        v = []
        for line in file:
            v.append(float(line.strip()))
        return v


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Requires true and prediction files as arguments!")
        exit(1)

    true = load_true_file(sys.argv[1])
    pred = load_pred_file(sys.argv[2])

    inv_ps = None
    if len(sys.argv) > 3:
        inv_ps = load_inv_ps_file(sys.argv[3])

    max_k = 10

    r = precision_at_k(true, pred, k=max_k)
    for k in range(max_k):
        print("P@{}: {}".format(k + 1, r[k]))

    r = recall_at_k(true, pred, k=max_k)
    for k in range(max_k):
        print("R@{}: {}".format(k + 1, r[k]))

    r = coverage_at_k(true, pred, k=max_k)
    for k in range(max_k):
        print("C@{}: {}".format(k + 1, r[k]))

    r = ndcg_at_k(true, pred, k=max_k)
    for k in range(max_k):
        print("nDCG@{}: {}".format(k + 1, r[k]))

    if inv_ps is not None:
        r = psprecision_at_k(true, pred, inv_ps=inv_ps, k=max_k)
        for k in range(max_k):
            print("PSP@{}: {}".format(k + 1, r[k]))
