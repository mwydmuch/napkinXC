#!/usr/bin/env python

import sys
import json

def get_mapping(files_in):
    labels_map = {}
    features_map = {0: 0}
    for file_in in files_in:
        with open(file_in) as fi:
            for line in fi:
                line = line.strip().split(" ")
                labels_map[int(line[0].strip(", "))] = 0 #TODO: support for multi-label
                features = [f.split(":") for f in line[1:]]
                for f in features:
                    if(len(f) > 1):
                        features_map[int(f[0])] = 0

    for i, l in enumerate(sorted(labels_map.keys())):
        labels_map[l] = str(i)
    print("Max idx of label in original mapping:", l)
    print("Max idx of label in remapped:", i)

    for i, f in enumerate(sorted(features_map.keys())):
        features_map[f] = str(i)
    print("Max idx of feature in original mapping:", f)
    print("Max idx of feature in remapped:", i)

    return labels_map, features_map


def remap(file_in, file_out, labels_map, features_map):
    with open(file_in) as fi:
        with open(file_out, "w") as fo:
            for line in fi:
                line = line.strip().split(" ")
                new_label = labels_map[int(line[0].strip(", "))]
                features = [f.split(":") for f in line[1:]]
                if features[0][0] == "0": # Skip feature 0
                    features = features[1:]
                new_features = [str(features_map[int(f[0])]) + ":" + f[1] for f in features if len(f) > 1]
                fo.write("{} {}\n".format(new_label, " ".join(new_features)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("The script requires 2 file paths as argument!")

    train_file_in = sys.argv[1]
    test_file_in = sys.argv[2]

    labels_map, features_map = get_mapping([train_file_in, test_file_in])

    remap(train_file_in, train_file_in + ".remapped", labels_map, features_map)
    remap(test_file_in, test_file_in + ".remapped", labels_map, features_map)

    with open(train_file_in + ".labels_map", 'w') as outfile:
        json.dump(labels_map, outfile)

    with open(train_file_in + ".features_map", 'w') as outfile:
        json.dump(features_map, outfile)
