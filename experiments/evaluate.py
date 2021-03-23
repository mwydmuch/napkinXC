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

    precision = 6
    max_k = 5

    measures = {
        "P": {"func": precision_at_k, "inv_ps": False},
        "R": {"func": recall_at_k, "inv_ps": False},
        "nDCG": {"func": ndcg_at_k, "inv_ps": False},
        "PSP": {"func": psprecision_at_k, "inv_ps": True},
        "PSR": {"func": psrecall_at_k, "inv_ps": True},
        "PSnDCG": {"func": psndcg_at_k, "inv_ps": True}
    }

    for m, v in measures.items():
        r = None
        if v["inv_ps"] and inv_ps is not None:
            r = v["func"](true, pred, inv_ps, k=max_k)
        else:
            r = v["func"](true, pred, k=max_k)
        if r is not None:
            for k in range(max_k):
                print(("{}@{}: {:." + str(precision) + "f}").format(m, k + 1, r[k]))
