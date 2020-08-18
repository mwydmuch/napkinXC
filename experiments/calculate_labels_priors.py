#!/usr/bin/env python3

import sys
from sklearn.datasets import load_svmlight_file


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: calculate_labels_priors.py [input file] [output file] [min value (optional)] [max value (optional)]")
        exit(1)

    input = sys.argv[1]
    output = sys.argv[2]
    min_value = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    max_value = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    X, y = load_svmlight_file(input, multilabel=True)
    y_count = {}
    row_count = 0
    for y_row in y:
        for y_i in y_row:
            y_count[y_i] = y_count.get(y_i, 0) + 1
        row_count += 1

    with open(output, "w") as file_out:
        for y_i in range(int(max(y_count, key=int)) + 1):
            file_out.write("{}\n".format(max(min(y_count.get(y_i, 0) / row_count, max_value), min_value)))

    seen_labels = sum(y_count.values())
    print("{} labels seen in {} rows ({:.3f} labels / row).".format(seen_labels, row_count, seen_labels / row_count))

