from napkinxc.metrics import *
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score as skl_f1_score


# Test datasets
T1 = [
    [1, 2, 3],
    [3, 4, 7]
]

L1l = [
    [1, 2, 4],
    [7, 6, 3]
]

L1l2 = [
    [1, 2],
    [7, 6, 3, 1]
]

L1lf = [
    [(1, 1.0), (2, 0.9), (4, 0.8)],
    [(7, 1.0), (6, 0.9), (3, 0.8)]
]

T2 = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1]
])

L2f = np.array([
    [0, 1, 0.9, 0.4, 0.8, 0.35, 0.3, 0.2],
    [0, 0.1, 0.3, 0.8, 0, 0.2, 0.9, 1.0]
])

L2b = np.array([
    [0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1]
])

T3 = csr_matrix((
    np.array([1, 1, 1, 1, 1, 1]),
    np.array([1, 2, 3, 3, 4, 7]),
    np.array([0, 3, 6]),
))

L3f = csr_matrix((
    np.array([1.0, 0.9, 0.4, 0.8, 0.85, 0.3, 0.95, 1]),
    np.array([1, 2, 3, 4, 3, 5, 6, 7]),
    np.array([0, 4, 8]),
))

L3b = csr_matrix((
    np.array([1, 1, 1, 1, 1, 1]),
    np.array([1, 2, 4, 3, 6, 7]),
    np.array([0, 3, 6]),
))

L4 = csr_matrix((
    np.array([1.0, 0.95, 0.8, 0.7, 1.0, 0.95, 0.8, 0.7]),
    np.array([4, 3, 2, 1, 6, 5, 4, 7]),
    np.array([0, 4, 8]),
))

T5 = [
    ["cat", "dog", "tiger"],
    ["tiger", "wolf", "kitty"]
]

L5l = [
    ["cat", "dog", "wolf"],
    ["kitty", "puppy", "tiger"]
]

# Sets of datasets
binary_set = [(T1, L1l), (T1, L1lf), (T2, L2b), (T3, L3b), (T5, L5l)]
ranking_set = [(T1, L1l), (T1, L1lf), (T1, L1l2), (T2, L2f), (T3, L3f), (T5, L5l)]


def _test_using_set(func, true_result, test_set):
    k = len(true_result)
    for (T, L) in test_set:
        func_result = func(T, L, k=k)
        assert np.allclose(true_result, func_result), f"{func.__name__}({T}, {L}, k={k}) = {func_result} != {true_result}"
        

def test_precision_at_k():
    true_p_at_3 = np.array([1, 3/4, 2/3])
    _test_using_set(precision_at_k, true_p_at_3, ranking_set)


def test_recall_at_k():
    true_r_at_3 = np.array([1/3, 1/2, 2/3])
    _test_using_set(recall_at_k, true_r_at_3, ranking_set)


def test_ndcg_at_k():
    true_ndcg_at_3 = np.array([1, 0.8065736, 0.73463936])
    _test_using_set(ndcg_at_k, true_ndcg_at_3, ranking_set)


def test_hamming_loss():
    true_hl = 2
    for (T, L) in binary_set:
        assert true_hl == hamming_loss(T, L), f"hamming_loss({T}, {L}) != {true_hl}"


def test_coverage_at_k():
    true_c_at_3 = np.array([2/5, 3/5, 4/5])
    _test_using_set(coverage_at_k, true_c_at_3, ranking_set)


def test_abandonment_at_k():
    true_a_at_3 = np.array([1, 1, 1])
    _test_using_set(abandonment_at_k, true_a_at_3, ranking_set)


def test_f1_measure():
    true_macro_f1_zdiv_0 = skl_f1_score(T2, L2b, average='macro', zero_division=0)
    true_macro_f1_zdiv_1 = skl_f1_score(T2, L2b, average='macro', zero_division=1)
    true_micro_f1 = skl_f1_score(T2, L2b, average='micro')
    true_samples_f1 = skl_f1_score(T2, L2b, average='samples')
    for (T, L) in binary_set[:-1]:  # all binary minus text examples
        assert true_macro_f1_zdiv_0 == f1_measure(T, L, average='macro', zero_division=0), f"f1_measure({T}, {L}, 'macro', 0) != {true_macro_f1_zdiv_0}"
        assert true_macro_f1_zdiv_1 == f1_measure(T, L, average='macro', zero_division=1), f"f1_measure({T}, {L}, 'macro', 1) != {true_macro_f1_zdiv_1}"
        assert true_micro_f1 == f1_measure(T, L, average='micro'), f"f1_measure({T}, {L}, 'micro') != {true_micro_f1}"
        assert true_samples_f1 == f1_measure(T, L, average='samples'), f"f1_measure({T}, {L}, 'samples') != {true_samples_f1}"
