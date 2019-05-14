#!/usr/bin/env python

import sys
import json

if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise ValueError("The script requires 1 dir paths as argument!")

    dir = sys.argv[1]

    with open(dir + "/Dmoz.train.labels_map") as json_file:
        labels_map = json.load(json_file)

    nodes_map = {"0": 0}
    child2parent_map = {}
    parents_set = set()
    parents_set.add("0")

    with open(dir + "/cat_hier.txt") as fi:
        for line in fi:
            nodes = line.strip().split(" ")
            child2parent_map[nodes[1]] = nodes[0]
            parents_set.add(nodes[0])
            for n in nodes:
                if n not in nodes_map:
                    nodes_map[n] = len(nodes_map)

    for p in list(child2parent_map.values()):
        if p not in child2parent_map and p != 0:
            child2parent_map[p] = '0'

    k = 0
    for c in child2parent_map.keys():
        if c not in parents_set:
            k += 1

    with open(dir + "/Dmoz.hier", "w") as fo:
        fo.write("{} {}\n".format(k, len(nodes_map)))
        for c, p in sorted(child2parent_map.items(), key=lambda x: x[1]):
            if c in parents_set:
                fo.write("{} {}\n".format(nodes_map[p], nodes_map[c]))
            else:
                if c in labels_map:
                    fo.write("{} {} {}\n".format(nodes_map[p], nodes_map[c], labels_map[c]))
