#!/usr/bin/env python

import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
napkinxc_path = os.path.join(file_dir, "../python")
sys.path.extend([file_dir, napkinxc_path])

from scripts_utils import *
from napkinxc.measures import *


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: evaluate.py [true file] [prediction file] [inverse propensities file (optional)]")
        exit(1)

    true = load_true_file(sys.argv[1])
    pred = load_pred_file(sys.argv[2], sort=True)

    inv_ps = None
    if len(sys.argv) > 3:
        inv_ps = load_weights_file(sys.argv[3])

    precision = 6
    max_k = 5

    measures = {
        #"HL": {"func": hamming_loss, "inv_ps": False},
        "P": {"func": precision_at_k, "inv_ps": False},
        "A": {"func": abandonment_at_k, "inv_ps": False},
        "R": {"func": recall_at_k, "inv_ps": False},
        # "nDCG": {"func": ndcg_at_k, "inv_ps": False},
        "PSP": {"func": psprecision_at_k, "inv_ps": True},
        # "PSnDCG": {"func": psndcg_at_k, "inv_ps": True},
        "MacP": {"func": macro_precision_at_k, "inv_ps": False},
        "MacF1": {"func": macro_f1_measure_at_k, "inv_ps": False},
        "C": {"func": coverage_at_k, "inv_ps": False},
        "MacR": {"func": macro_recall_at_k, "inv_ps": False}, 
    }

    for m, v in measures.items():
        r = None
        if v["inv_ps"]:
            if  inv_ps is not None:
                r = v["func"](true, pred, inv_ps, k=max_k)
        else:
            r = v["func"](true, pred, k=max_k)
        if r is not None:
            for k in range(max_k):
                print(("{}@{}: {:." + str(precision) + "f}").format(m, k + 1, r[k]))
