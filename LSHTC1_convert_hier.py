#!/usr/bin/env python

import sys
import json

if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise ValueError("The script requires 1 dir paths as argument!")

    dir = sys.argv[1]

    with open(dir + "/LSHTC1.train.labels_map") as json_file:
        labels_map = json.load(json_file)

    labels_paths = {}
    nodes_map = {}
    edges_set = set()

    with open(dir + "/cat_hier.txt") as fi:
        for line in fi:
            path = ["0"] + line.strip().split(" ")
            labels_paths[path[-1]] = path
            for p in path:
                if p not in nodes_map:
                    nodes_map[p] = len(nodes_map)

    with open(dir + "/LSHTC1.hier", "w") as fo:
        fo.write("{} {}\n".format(len(labels_paths), len(nodes_map)))
        for l, p in labels_paths.items():
            for i in range(len(p) - 2):
                if (p[i], p[i + 1]) not in edges_set:
                    fo.write("{} {}\n".format(nodes_map[p[i]], nodes_map[p[i + 1]]))
                    edges_set.add((p[i], p[i + 1]))
            if l in labels_map:
                fo.write("{} {} {}\n".format(nodes_map[p[-2]], nodes_map[p[-1]], labels_map[l]))
