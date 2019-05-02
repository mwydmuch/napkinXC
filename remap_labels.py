#!/usr/bin/env python

import sys


def reorder_labels(file_in, file_out, labels_map):
    with open(file_in) as fi:
        with open(file_out, "w") as fo:
            for line in fi:
                line = line.split(" ")
                label = line[0]
                if label in labels_map:
                    new_label = labels_map[label]
                else:
                    new_label = len(labels_map)
                    labels_map[label] = new_label

                fo.write("{} {}".format(new_label, " ".join(line[1:])))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("The script requires 4 file paths as argument!")

    train_file_in = sys.argv[1]
    train_file_out = sys.argv[2]
    test_file_in = sys.argv[3]
    test_file_out = sys.argv[4]

    labels_map = {}

    reorder_labels(train_file_in, train_file_out, labels_map)
    reorder_labels(test_file_in, test_file_out, labels_map)
