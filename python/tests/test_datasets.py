from napkinxc.datasets import *
import os
from shutil import rmtree


def test_load_bow():
    for d in DATASETS:
        if 'bow' in d['formats']:
            X, Y = load_dataset(d['name'], subset='train', format='bow', root='./data_tmp')
            rmtree(os.path.basename(d['name']['bow']['train']))
