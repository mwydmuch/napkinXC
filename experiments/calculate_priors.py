#!/usr/bin/env python3

import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
napkinxc_path = os.path.join(file_dir, "../python")
sys.path.extend([file_dir, napkinxc_path])

from scripts_utils import *
from napkinxc.measures import labels_priors


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: calculate_priors.py [input file] [output file]")
        exit(1)

    true_file = sys.argv[1]
    true = load_true_file(sys.argv[1])

    priors = labels_priors(true)
    with open(sys.argv[2], "w") as out:
        for p in priors:
            out.write("{}\n".format(p))

