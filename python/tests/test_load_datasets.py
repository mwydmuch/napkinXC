from napkinxc.datasets import *


def test_load_bow():
    for d in DATASETS.values():
        if 'bow' in d['formats']:
            for s in d['subsets']:
                print(d['name'], s)
                X, Y = load_dataset(d['name'], subset=s, format='bow')
