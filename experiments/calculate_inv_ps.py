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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Requires true file and output as arguments!")
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
