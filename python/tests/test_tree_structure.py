import shutil
from napkinxc.datasets import load_dataset
from napkinxc.models import PLT

from conf import *
MODEL_PATH = get_model_path(__file__)


def test_set_get_tree_structure():
    X, Y = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    plt = PLT(MODEL_PATH)
    plt.build_tree(X, Y)
    tree_structure = plt.get_tree_structure()
    plt.set_tree_structure(tree_structure)
    tree_structure2 = plt.get_tree_structure()
    assert len(set(tree_structure) - set(tree_structure2)) == 0

    nodes_to_update = plt.get_nodes_to_update(Y)
    assert len(nodes_to_update) == X.shape[0]

    nodes_updates = plt.get_nodes_updates(Y)
    assert len(nodes_updates) == len(tree_structure)

    plt.fit(X, Y)
    tree_structure3 = plt.get_tree_structure()
    assert len(set(tree_structure) - set(tree_structure3)) == 0

    shutil.rmtree(MODEL_PATH, ignore_errors=True)


def test_build_tree_structure_reproducibility():
    X, Y = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    plt = PLT(MODEL_PATH + "-1", seed=1993)
    plt.build_tree(X, Y)
    tree_structure = plt.get_tree_structure()

    plt2 = PLT(MODEL_PATH + "-2", seed=1993)
    plt2.build_tree(X, Y)
    tree_structure2 = plt2.get_tree_structure()

    assert len(set(tree_structure) - set(tree_structure2)) == 0

    shutil.rmtree(MODEL_PATH + "-1", ignore_errors=True)
    shutil.rmtree(MODEL_PATH + "-2", ignore_errors=True)
