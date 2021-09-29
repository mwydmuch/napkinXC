import shutil
import os
from napkinxc.datasets import load_dataset
from napkinxc.models import BR, PLT
from napkinxc.measures import precision_at_k


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-eurlex-model")
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-data")
repeat = 10

def test_prediction_reproducibility():
    X_train, Y_train = load_dataset("eurlex-4k", "train", root=data_path)
    X_test, Y_test = load_dataset("eurlex-4k", "test", root=data_path)

    plt = PLT(model_path)
    plt.fit(X_train, Y_train)
    Y_pred = plt.predict(X_test, top_k=1)
    p_at_1 = precision_at_k(Y_test, Y_pred, k=1)

    for _ in range(repeat):
        plt = PLT(model_path)
        Y_pred = plt.predict(X_test, top_k=1)
        assert p_at_1 == precision_at_k(Y_test, Y_pred, k=1)

    shutil.rmtree(model_path, ignore_errors=True)


def test_seed_reproducibility():
    X_train, Y_train = load_dataset("eurlex-4k", "train", root=data_path)
    X_test, Y_test = load_dataset("eurlex-4k", "test", root=data_path)

    for i in range(repeat):
        plt_1 = PLT(model_path + "-1", seed=i)
        plt_1.fit(X_train, Y_train)
        Y_pred_1 = plt_1.predict(X_test, top_k=1)
        p_at_1_1 = precision_at_k(Y_test, Y_pred_1, k=1)

        plt_2 = PLT(model_path + "-2", seed=i)
        plt_2.fit(X_train, Y_train)
        Y_pred_2 = plt_2.predict(X_test, top_k=1)
        p_at_1_2 = precision_at_k(Y_test, Y_pred_2, k=1)

        assert p_at_1_1 == p_at_1_2
        shutil.rmtree(model_path + "-1", ignore_errors=True)
        shutil.rmtree(model_path + "-2", ignore_errors=True)
