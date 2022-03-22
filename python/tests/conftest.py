import shutil
from napkinxc.datasets import download_dataset

from conf import *


def pytest_configure(config):
    print("Downloading/checking test data...")
    download_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    download_dataset(TEST_DATASET, "test", root=TEST_DATA_PATH)


def pytest_unconfigure(config):
    if REMOVE_TEST_DATA:
        print("Removing test data...")
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)
