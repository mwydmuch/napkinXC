import os

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
TEST_DATASET = "yeast"  # old: "eurlex-4k"
TEST_TRAIN_DATA_PATH = os.path.join(TEST_DATA_PATH, f"{TEST_DATASET}/{TEST_DATASET}_train.txt")
TEST_TEST_DATA_PATH = os.path.join(TEST_DATA_PATH, f"{TEST_DATASET}/{TEST_DATASET}_test.txt")
REMOVE_TEST_DATA = False
TEST_SEED = 1993
SCORE_RANGE = [0.61, 0.77]  # old for eurlex-4k: [0.72, 0.82]


def get_model_path(test_file):
    return os.path.join(os.path.dirname(os.path.abspath(test_file)), f"{os.path.basename(test_file)}_model")
