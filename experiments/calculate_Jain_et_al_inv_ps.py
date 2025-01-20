#!/usr/bin/env python

import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
napkinxc_path = os.path.join(file_dir, "../python")
sys.path.extend([file_dir, napkinxc_path])

from scripts_utils import *
from napkinxc.metrics import Jain_et_al_inverse_propensity


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: calculate_Jain_et_al_inv_ps.py [input file] [output file]")
        exit(1)

    true_file = sys.argv[1]
    true = load_true_file(sys.argv[1])

    A = 0.55
    B = 1.5

    if '/wikiLSHTC/' in true_file or '/WikipediaLarge-500K/' in true_file:
        A = 0.5
        B = 0.4
    elif '/amazon/' in true_file or '/amazon-3M/' in true_file:
        A = 0.6
        B = 2.6

    inv_ps = Jain_et_al_inverse_propensity(true, A=A, B=B)
    with open(sys.argv[2], "w") as out:
        for ip in inv_ps:
            out.write("{}\n".format(ip))
