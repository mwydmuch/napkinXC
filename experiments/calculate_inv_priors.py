#!/usr/bin/env python3

import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
napkinxc_path = os.path.join(file_dir, "../python")
sys.path.extend([file_dir, napkinxc_path])

from scripts_utils import *
from napkinxc.metrics import inverse_labels_priors


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: calculate_inv_priors.py [input file] [output file]")
        exit(1)

    true_file = sys.argv[1]
    true = load_true_file(sys.argv[1])

    inv_priors = inverse_labels_priors(true)
    with open(sys.argv[2], "w") as out:
        for ip in inv_priors:
            out.write("{}\n".format(ip))

