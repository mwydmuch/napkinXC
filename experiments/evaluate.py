#!/usr/bin/env python

import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
napkinxc_path = os.path.join(file_dir, "../python")
sys.path.extend([file_dir, napkinxc_path])

from scripts_utils import *
from napkinxc.metrics import *


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: evaluate.py [true file] [prediction file] [inverse propensities/labels weights file (optional)]")
        exit(1)

    true = load_true_file(sys.argv[1])
    pred = load_pred_file(sys.argv[2])

    labels_weights = None
    if len(sys.argv) > 3:
        labels_weights = load_weights_file(sys.argv[3])

    precision = 6
    max_k = 5

    metrics = {
        #"HL": {"func": hamming_loss, "needs_weights": False},
        "P": {"func": precision_at_k, "needs_weights": False},
        "R": {"func": recall_at_k, "needs_weights": False},
        "nDCG": {"func": ndcg_at_k, "needs_weights": False},
        "PSP": {"func": psprecision_at_k, "needs_weights": True},
        "PSnDCG": {"func": psndcg_at_k, "needs_weights": True},
        "mP": {"func": macro_precision_at_k, "needs_weights": False},
        "mR": {"func": macro_recall_at_k, "needs_weights": False},
        "mF1": {"func": macro_f1_measure_at_k, "needs_weights": False},
        "C": {"func": coverage_at_k, "needs_weights": False},
    }

    for m, v in metrics.items():
        r = None
        if v["needs_weights"]:
            if labels_weights is not None:
                r = v["func"](true, pred, labels_weights, k=max_k)
        else:
            r = v["func"](true, pred, k=max_k)
        if r is not None:
            for k in range(max_k):
                print(("{}@{}: {:." + str(precision) + "f}").format(m, k + 1, r[k]))
