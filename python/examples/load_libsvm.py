#!/usr/bin/env python3

# This demo shows how to load libsvm file using napkinXC's load_libsvm_file function,
# which is easier to use, faster, and more memory efficient than Sklearn's load_svmlight_file.
# This examples requires Sklearn installed.

from time import time
from sklearn.datasets import load_svmlight_file
from napkinxc.datasets import download_dataset, load_libsvm_file

# Use download_dataset function to download one of the benchmark datasets
# from XML Repository (http://manikvarma.org/downloads/XC/XMLRepository.html).
download_dataset("eurlex-4k", "train")
file = "data/Eurlex/eurlex_train.txt"

# Load using Sklearn
# Because Sklearn method cannot handel header from XML Repository, offset and number of features needs to be provided.
start = time()
X, Y = load_svmlight_file(file, multilabel=True, zero_based=True, n_features=5000, offset=1)
print("Sklearn's load_svmlight_file time:", time() - start)

# Load using napkinXC
start = time()
X, Y = load_libsvm_file(file)
print("napkinXC's load_libsvm_file time:", time() - start)
