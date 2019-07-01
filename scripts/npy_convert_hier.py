#!/usr/bin/env python

import numpy as np
import pathlib
from tqdm import tqdm


def convert_hier(infile, outfile):
    edges_set = set()

    with open(infile) as fi:
        nodes = eval("{}".format(fi.read()))
        leaves = nodes[0]

        with open(outfile, "w") as fo:
            fo.write("{} {}\n".format(len(leaves), len(nodes)))
            for l in leaves:
                p = []
                for i in range(len(nodes)):
                    if l in nodes[i]:
                        p.append(i)

                for i in range(len(p) - 2):
                    if (p[i], p[i + 1]) not in edges_set:
                        fo.write("{} {}\n".format(p[i], p[i + 1]))
                    edges_set.add((p[i], p[i + 1]))

                fo.write("{} {} {}\n".format(p[-2], p[-1], l - 1))


if __name__ == "__main__":
    datasets = ['CAL101', 'CAL256', 'VOC2006', 'VOC2007', 'PROTEIN2']

    for d in datasets:
        print("Converting {} ...".format(d))
        convert_hier("npy_data/hierarchy/{}/hierarchy_full.txt".format(d), "data/HIERARCHICAL_PREDEFINED_{}/HIERARCHICAL_PREDEFINED_{}.hier".format(d, d))
        convert_hier("npy_data/hierarchy/{}/hierarchy_random.txt".format(d), "data/HIERARCHICAL_RANDOM_{}/HIERARCHICAL_RANDOM_{}.hier".format(d, d))

