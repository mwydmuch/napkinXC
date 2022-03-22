import shutil
from napkinxc.datasets import load_dataset
from napkinxc.models import *
from napkinxc.measures import precision_at_k

from conf import *
MODEL_PATH = get_model_path(__file__)


def _test_model(model_class, model_config):
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    X_test, Y_test = load_dataset(TEST_DATASET, "test", root=TEST_DATA_PATH)

    model = model_class(MODEL_PATH, seed=TEST_SEED, **model_config)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test, top_k=1)
    p_at_1 = precision_at_k(Y_test, Y_pred, k=1)

    assert SCORE_RANGE[0] < p_at_1 < SCORE_RANGE[1]

    shutil.rmtree(MODEL_PATH, ignore_errors=True)


def test_plt_train_test():
    _test_model(PLT, {})


def test_hsm_train_test():
    _test_model(HSM, {"pick_one_label_weighting": True})


def test_br_train_test():
    _test_model(BR, {"optimizer": "adagrad", "epochs": 1})


def test_ovr_train_test():
    _test_model(OVR, {"pick_one_label_weighting": True})
