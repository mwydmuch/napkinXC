from time import time
import numpy as np
import shutil

# pip install git+https://github.com/kunaldahiya/pyxclib.git
from xclib.evaluation.xc_metrics import *

from napkinxc.datasets import load_dataset, to_csr_matrix
from napkinxc.models import PLT
from napkinxc.measures import *


from conf import *
MODEL_PATH = get_model_path(__file__)


def test_compare_napkinxc_with_xclib():
    k = 5
    
    # Train model and predict
    X_train, Y_train = load_dataset(TEST_DATASET, "train", root=TEST_DATA_PATH)
    X_test, Y_test = load_dataset(TEST_DATASET, "test", root=TEST_DATA_PATH)
    plt = PLT(MODEL_PATH)
    plt.fit(X_train, Y_train)
    Y_pred = plt.predict_proba(X_test, top_k=k)
    shutil.rmtree(MODEL_PATH, ignore_errors=True)

    # Prepare dataset
    csr_Y_train = to_csr_matrix(Y_train)
    csr_Y_test = to_csr_matrix(Y_test)
    csr_Y_pred = to_csr_matrix(Y_pred, shape=csr_Y_test.shape)

    # Calculate propensities
    nxc_inv_ps = inverse_propensity(Y_train, A=0.55, B=1.5)
    csr_nxc_inv_ps = inverse_propensity(csr_Y_train, A=0.55, B=1.5)
    xcl_inv_ps = compute_inv_propesity(csr_Y_train, A=0.55, B=1.5)
    assert np.allclose(nxc_inv_ps, csr_nxc_inv_ps)
    assert np.allclose(nxc_inv_ps, xcl_inv_ps)

    # Compare results
    measures = {
        "P@k": {"nxc": precision_at_k, "xclib": precision, "inv_ps": False},
        "R@k": {"nxc": recall_at_k, "xclib": recall, "inv_ps": False},
        "nDCG@k": {"nxc": ndcg_at_k, "xclib": ndcg, "inv_ps": False},
        "PSP@k": {"nxc": psprecision_at_k, "xclib": psprecision, "inv_ps": True},
        "PSR@k": {"nxc": psrecall_at_k, "xclib": psrecall, "inv_ps": True},
        "PSnDCG@k": {"nxc": psndcg_at_k, "xclib": psndcg, "inv_ps": True}
    }

    print("\n")
    for m, v in measures.items():
        print("\n{} time comparison:".format(m))
        t_start = time()
        xclib_r = v["xclib"](csr_Y_pred, csr_Y_test, xcl_inv_ps, k=k) if v["inv_ps"] else v["xclib"](csr_Y_pred, csr_Y_test, k=k)
        print("\txclib.evaluation.xc_metrics.{} with csr_matrices: {}s".format(v["xclib"].__name__, time() - t_start))

        t_start = time()
        nxc_r = v["nxc"](Y_test, Y_pred, xcl_inv_ps, k=k) if v["inv_ps"] else v["nxc"](Y_test, Y_pred, k=k)
        print("\tnapkinXC.measures.{} with lists: {}s".format(v["nxc"].__name__, time() - t_start))

        t_start = time()
        csr_nxc_r = v["nxc"](csr_Y_test, csr_Y_pred, csr_nxc_inv_ps, k=k) if v["inv_ps"] else v["nxc"](csr_Y_test, csr_Y_pred, k=k)
        print("\tnapkinXC.measures.{} with csr_matrices: {}s".format(v["nxc"].__name__, time() - t_start))

        assert np.allclose(nxc_r, csr_nxc_r)
        assert np.allclose(nxc_r, xclib_r)
