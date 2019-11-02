import sys
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


def remap_labels(y, mapping):
    for i in range(y.shape[0]):
        new_id = mapping.get(y[i], len(mapping))
        if new_id == len(mapping):
            mapping[y[i]] = new_id
        y[i] = new_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("The script requires 2 file paths as argument!")

    train_file_in = sys.argv[1]
    test_file_in = sys.argv[2]

    mapping = {}
    X, y = load_svmlight_file(train_file_in)
    remap_labels(y, mapping)
    dump_svmlight_file(X, y, train_file_in + ".remapped")

    X, y = load_svmlight_file(test_file_in)
    remap_labels(y, mapping)
    dump_svmlight_file(X, y, test_file_in + ".remapped")


    print("Max id of label in original mapping:", max(mapping, key=int))
    print("Max id of label in remapped:", len(mapping))
