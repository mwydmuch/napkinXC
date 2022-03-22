import pytest
import os
import shutil

from napkinxc.datasets import download_dataset


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
train_path = os.path.join(data_path, "Eurlex/eurlex_train.txt")
test_path = os.path.join(data_path, "Eurlex/eurlex_test.txt")


def pytest_configure(config):
    print("Downloading test data...")
    download_dataset("eurlex-4k", "train", root=data_path)
    download_dataset("eurlex-4k", "test", root=data_path)


def pytest_unconfigure(config):
    print("Removing test data...")
    shutil.rmtree(data_path, ignore_errors=True)
