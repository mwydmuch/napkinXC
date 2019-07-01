#!/usr/bin/env python

import numpy as np
import pathlib
from tqdm import tqdm


def np2dense(a, name, ext):
    pathlib.Path("data/{}".format(name)).mkdir(parents=True, exist_ok=True)
    rows, cols = a.shape
    with open("data/{}/{}.{}".format(name, name, ext), "w") as out:
        for i in tqdm(range(rows)):
            for j in range(cols):
                if j == 0:
                    out.write("{}".format(int(a[i, j]) - 1))
                else:
                    out.write(" {}".format(a[i, j]))
            out.write("\n")


def np2libsvm(a, name, ext):
    pathlib.Path("data/{}".format(name)).mkdir(parents=True, exist_ok=True)
    rows, cols = a.shape
    with open("data/{}/{}.{}".format(name, name, ext), "w") as out:
        for i in tqdm(range(rows)):
            for j in range(cols):
                if j == 0:
                    out.write("{}".format(int(a[i, j]) - 1))
                elif a[i, j] != 0:
                    out.write(" {}:{}".format(j, a[i, j]))
            out.write("\n")


if __name__ == "__main__":
    datasets = ['CAL101', 'CAL256', 'VOC2006', 'VOC2007', 'PROTEIN2']
    prefix = ['FLAT', 'HIERARCHICAL_PREDEFINED', 'HIERARCHICAL_RANDOM']

    for d in datasets:
        for p in prefix:
            print("Converting {}_{} ...".format(p, d))
            a = np.load("npy_data/{}_{}_TEST.npy".format(p, d))
            #np2dense(a, "{}_{}".format(p, d), "test")
            np2libsvm(a, "{}_{}".format(p, d), "test")

            a = np.load("npy_data/{}_{}_TRAINVAL.npy".format(p, d))
            #np2dense(a, "{}_{}".format(p, d), "train")
            np2libsvm(a, "{}_{}".format(p, d), "train")
