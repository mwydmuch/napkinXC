import shutil
import os
from napkinxc.datasets import load_dataset
from napkinxc.models import BR, PLT
from napkinxc.measures import precision_at_k


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{os.path.basename(__file__)}_model")
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


# Basic PLT test on eurlex
def test_eurlex_plt_train_test():
    X_train, Y_train = load_dataset("eurlex-4k", "train", root=data_path)
    X_test, Y_test = load_dataset("eurlex-4k", "test", root=data_path)
    plt = PLT(model_path, optimizer="adagrad", epochs=1)
    plt.fit(X_train, Y_train)
    Y_pred = plt.predict(X_test, top_k=1)
    p_at_1 = precision_at_k(Y_test, Y_pred, k=1)
    assert 0.78 < p_at_1 < 0.82
    shutil.rmtree(model_path, ignore_errors=True)


# Basic BR test on eurlex
def _test_eurlex_br_adagrad_train_test():
    X_train, Y_train = load_dataset("eurlex-4k", "train", root=data_path)
    X_test, Y_test = load_dataset("eurlex-4k", "test", root=data_path)
    br = BR(model_path, optimizer="adagrad", epochs=1)
    br.fit(X_train, Y_train)
    Y_pred = br.predict(X_test, top_k=1)
    p_at_1 = precision_at_k(Y_test, Y_pred, k=1)
    assert 0.78 < p_at_1 < 0.82
    shutil.rmtree(model_path, ignore_errors=True)
