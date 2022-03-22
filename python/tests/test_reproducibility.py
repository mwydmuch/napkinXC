import shutil
import os
from napkinxc.datasets import load_dataset
from napkinxc.models import BR, PLT
from napkinxc.measures import precision_at_k

from conf import *
MODEL_PATH = get_model_path(__file__)


repeat = 5
model_configs = [
    {"tree_search_type": "exact"},
    {"tree_search_type": "beam"},
    {"ensemble": 3, "tree_search_type": "exact"},
    {"ensemble": 3, "tree_search_type": "beam"},
]
representation_configs = [
    {"load_as": "sparse"},
    {"load_as": "map"},
    {"load_as": "dense"},
]


def test_plt_exact_prediction_reproducibility():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    X_test, Y_test = load_dataset(TEST_DATASET, "test", root=TEST_DATA_PATH)

    print("\n")
    for mc in model_configs:
        print("model config: ", mc)
        plt = PLT(MODEL_PATH, **mc)
        plt.fit(X_train, Y_train)
        Y_pred = plt.predict(X_test, top_k=1)
        p_at_1 = precision_at_k(Y_test, Y_pred, k=1)

        for rc in representation_configs:
            print("  prediction config: ", rc)
            for _ in range(repeat):
                plt = PLT(MODEL_PATH, **mc, **rc)
                Y_pred = plt.predict(X_test, top_k=1)
                assert p_at_1 == precision_at_k(Y_test, Y_pred, k=1)

        shutil.rmtree(MODEL_PATH, ignore_errors=True)


def test_seed_reproducibility():
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    X_test, Y_test = load_dataset(TEST_DATASET, "test", root=TEST_DATA_PATH)

    for i in range(repeat):
        plt_1 = PLT(MODEL_PATH + "-1", optimizer="adagrad", epochs=1, loss="log", seed=i)
        plt_1.fit(X_train, Y_train)
        Y_pred_1 = plt_1.predict(X_test, top_k=1)
        p_at_1_1 = precision_at_k(Y_test, Y_pred_1, k=1)
        tree_structure_1 = plt_1.get_tree_structure()

        plt_2 = PLT(MODEL_PATH + "-2", optimizer="adagrad", epochs=1, loss="log", seed=i)
        plt_2.fit(X_train, Y_train)
        Y_pred_2 = plt_2.predict(X_test, top_k=1)
        p_at_1_2 = precision_at_k(Y_test, Y_pred_2, k=1)
        tree_structure_2 = plt_2.get_tree_structure()

        assert len(set(tree_structure_1) - set(tree_structure_2)) == 0
        assert p_at_1_1 == p_at_1_2

        shutil.rmtree(MODEL_PATH + "-1", ignore_errors=True)
        shutil.rmtree(MODEL_PATH + "-2", ignore_errors=True)
