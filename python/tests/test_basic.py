import shutil
import os
from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-eurlex-model")
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-data")


def test_eurlex_train_test():

    # Basic test on eurlex
    X_train, Y_train = load_dataset("eurlex-4k", "train", root=data_path)
    X_test, Y_test = load_dataset("eurlex-4k", "test", root=data_path)
    plt = PLT(model_path)
    plt.fit(X_train, Y_train)
    Y_pred = plt.predict(X_test, top_k=1)
    p_at_1 = precision_at_k(Y_test, Y_pred, k=1)
    assert 0.79 < p_at_1 < 0.82
    shutil.rmtree(model_path, ignore_errors=True)
