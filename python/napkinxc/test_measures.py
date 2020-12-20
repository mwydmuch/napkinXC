from measures import *
from scipy.sparse import csr_matrix
import numpy as np

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

print(L4[0].indices)


#print(len(T1), len(T2), len(T3))

print(precision_at_k(T1, L1l, 3))
print(precision_at_k(T1, L1lf, 3))
print(precision_at_k(T2, L2f, 3))
print(precision_at_k(T3, L3f, 3))

print(recall_at_k(T1, L1l, 3))
print(recall_at_k(T1, L1lf, 3))
print(recall_at_k(T2, L2f, 3))
print(recall_at_k(T3, L3f, 3))

print(ndcg_at_k(T1, L1l, 3))
print(ndcg_at_k(T1, L1lf, 3))
print(ndcg_at_k(T2, L2f, 3))
print(ndcg_at_k(T3, L3f, 3))

print(hamming_loss(T1, L1l))
print(hamming_loss(T1, L1lf))
print(hamming_loss(T2, L2b))
print(hamming_loss(T2, L2f > 0.5))
print(hamming_loss(T3, L3b))
print(hamming_loss(T3, L3f > 0.5))