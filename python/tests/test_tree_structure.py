import shutil
import os
from napkinxc.datasets import load_dataset
from napkinxc.models import PLT


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-eurlex-model")
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-data")


def test_tree_structure():
    X, Y = load_dataset("eurlex-4k", "train", root=data_path)
    plt = PLT(model_path)
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

    shutil.rmtree(model_path, ignore_errors=True)
