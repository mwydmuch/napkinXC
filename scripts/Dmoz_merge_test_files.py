#!/usr/bin/env python

import sys

if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise ValueError("The script requires 1 dir paths as argument!")

    dir = sys.argv[1]

    with open(dir + "/Dmoz.test_labels") as fi_l:
        with open(dir + "/Dmoz.test_features") as fi_f:
            with open(dir + "/Dmoz.test", "w") as fo:
                for l, f in zip(fi_l, fi_f):
                    fo.write("{} {}\n".format(l.strip(), " ".join(f.strip().split(" ")[1:])))
