from napkinxc.measures import *
from scipy.sparse import csr_matrix
import numpy as np
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

L1lf = [
    [(1, 1.0), (2, 0.9), (4, 0.8)],
    [(7, 1.0), (6, 0.9), (3, 0.8)]
]

T2 = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1]
])

L2f = np.array([
    [0, 1, 0.9, 0.4, 0.8, 0.3, 0.3, 0.2],
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
    np.array([1.0, 0.9, 0.4, 0.8, 0.8, 0.3, 0.9, 1]),
    np.array([1, 2, 3, 4, 3, 5, 6, 7]),
    np.array([0, 4, 8]),
))

L3b = csr_matrix((
    np.array([1, 1, 1, 1, 1, 1]),
    np.array([1, 2, 4, 3, 6, 7]),
    np.array([0, 3, 6]),
))

L4 = csr_matrix((
    np.array([1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7]),
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
ranking_set = binary_set + [(T2, L2f), (T3, L3f)]


def test_precision_at_k():
    true_p_at_3 = np.array([1, 3/4, 2/3])
    for (T, L) in ranking_set:
        assert np.array_equal(true_p_at_3, precision_at_k(T, L, 3)), "precision_at_k({}, {}, k=3) != {}".format(T, L, true_p_at_3)


def test_recall_at_k():
    true_r_at_3 = np.array([1/3, 1/2, 2/3])
    for (T, L) in ranking_set:
        assert np.array_equal(true_r_at_3, recall_at_k(T, L, 3)), "recall_at_k({}, {}, k=3) != {}".format(T, L, true_r_at_3)


def test_ndcg_at_k():
    true_ndcg_at_3 = np.array([1, 0.8065736, 0.73463936])
    for (T, L) in ranking_set:
        assert np.allclose(true_ndcg_at_3, ndcg_at_k(T, L, 3)), "ndcg_at_k({}, {}, k=3) != {}".format(T, L, true_ndcg_at_3)


def test_hamming_loss():
    true_hl = 2
    for (T, L) in binary_set:
        assert true_hl == hamming_loss(T, L), "hamming_loss({}, {}) != {}".format(T, L, true_hl)


def test_f1_measure():
    true_macro_f1_zdiv_0 = skl_f1_score(T2, L2b, average='macro', zero_division=0)
    true_macro_f1_zdiv_1 = skl_f1_score(T2, L2b, average='macro', zero_division=1)
    true_micro_f1 = skl_f1_score(T2, L2b, average='micro')
    true_samples_f1 = skl_f1_score(T2, L2b, average='samples')
    for (T, L) in binary_set[:-1]:  # all binary minus text examples
        assert true_macro_f1_zdiv_0 == f1_measure(T, L, average='macro', zero_division=0), "f1_measure({}, {}, 'macro', 0) != {}".format(T, L, true_macro_f1_zdiv_0)
        assert true_macro_f1_zdiv_1 == f1_measure(T, L, average='macro', zero_division=1), "f1_measure({}, {}, 'macro', 1) != {}".format(T, L, true_macro_f1_zdiv_1)
        assert true_micro_f1 == f1_measure(T, L, average='micro'), "f1_measure({}, {}, 'micro') != {}".format(T, L, true_micro_f1)
        assert true_samples_f1 == f1_measure(T, L, average='samples'), "f1_measure({}, {}, 'samples') != {}".format(T, L, true_samples_f1)
