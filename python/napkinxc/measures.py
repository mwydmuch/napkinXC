# Copyright (c) 2020 by Marek Wydmuch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from math import log2, log
import numpy as np
from scipy.sparse import csr_matrix


def precision_at_k(Y_true, Y_pred, k=5):
    """
    Calculate precision at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels
    :type Y_true: ndarray, csr_matrix, list[list[int]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as list of labels or tuples of labels with scores (idx, score)..
        In case of matrix, ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: ndarray with values of precision at 1-k places.
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    sum = np.zeros(k)
    count = 0
    for t, p in zip(Y_true, Y_pred):
        p_at_i = 0
        for i, p_i in enumerate(p[:k]):
            p_at_i += 1 if p_i in t else 0
            sum[i] += p_at_i / (i + 1)
        count += 1
    return sum / count


def recall_at_k(Y_true, Y_pred, k=5, zero_division=0):
    """
    Calculate recall at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels
    :type Y_true: ndarray, csr_matrix, list[list[int]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as list of labels or tuples of labels with scores (idx, score)..
        In case of matrix, ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param zero_division: Value to add when there is a zero division.
    :type zero_division: float {0, 1}
    :return: ndarray with values of recall at 1-k places.
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    sum = np.zeros(k)
    count = 0
    for t, p in zip(Y_true, Y_pred):
        if len(t) > 0:
            r_at_k = 0
            for i, p_i in enumerate(p[:k]):
                r_at_k += 1 if p_i in t else 0
                sum[i] += r_at_k / len(t)
        else:
            sum += zero_division
        count += 1
    return sum / count


def coverage_at_k(Y_true, Y_pred, k=5):
    """
    Calculate coverage at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels
    :type Y_true: ndarray, csr_matrix, list[list[int]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as list of labels or tuples of labels with scores (idx, score)..
        In case of matrix, ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: ndarray with values of coverage at 1-k places.
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    uniq_t = [set() for _ in range(k)]
    uniq_tp = [set() for _ in range(k)]
    for t, p in zip(Y_true, Y_pred):
        for t_i in t:
            uniq_t.add(t_i)
        for p_i in p[:k]:
            if p_i in t:
                uniq_tp.add(p_i)
    return len(uniq_tp) / len(uniq_t)


def dcg_at_k(Y_true, Y_pred, k=5):
    """
    Calculate DCG at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels
    :type Y_true: ndarray, csr_matrix, list[list[int]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as list of labels or tuples of labels with scores (idx, score)..
        In case of matrix, ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: ndarray with values of DCG at 1-k places.
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    sum = np.zeros(k)
    count = 0
    for t, p in zip(Y_true, Y_pred):
        dcg_at_i = 0
        for i, p_i in enumerate(p[:k]):
            dcg_at_i += 1 / log2(i + 2) if p_i in t else 0
            sum[i] += dcg_at_i
        count += 1
    return sum / count


def ndcg_at_k(Y_true, Y_pred, k=5):
    """
    Calculate nDCG at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels
    :type Y_true: ndarray, csr_matrix, list[list[int]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as list of labels or tuples of labels with scores (idx, score)..
        In case of matrix, ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: ndarray with values of nDCG at 1-k places.
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    sum = np.zeros(k)
    count = 0
    for t, p in zip(Y_true, Y_pred):
        dcg_at_i = 0
        norm_at_i = 0
        for i, p_i in enumerate(p[:k]):
            norm_at_i += 1 / log2(i + 2)
            dcg_at_i += 1 / log2(i + 2) if p_i in t else 0
            sum[i] += dcg_at_i / norm_at_i
        count += 1
    return sum / count


def hamming_loss(Y_true, Y_pred):
    """
    Calculate hamming loss.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels
    :type Y_true: ndarray, csr_matrix, list[list[int]]
    :param Y_pred: Predicted labels provided as a matrix with scores or list of list of labels or tuples of labels with scores (idx, score).
    :type Y_pred: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
    :return: ndarray with values of nDCG at 1-k places.
    """
    
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred)

    sum = 0
    count = 0
    for t, p in zip(Y_true, Y_pred):
        sum += len(p) + len(t) - 2 * len(set(t).intersection(p))
        count += 1

    return sum / count


def inverse_propensity(Y, A=0.55, B=1.5):
    """
    Calculate inverse propensity as proposed in Jain et al. 2016.

    :param Y: Labels (typically ground truth for train data) provided as a matrix with non-zero values for relevant labels
    :type Y: ndarray, csr_matrix, list of lists of ints or tuples (idx, score)
    :param A: A value, typical values:

        - 0.5: ``WikiLSHTC-325K``
        - 0.6: ``Amazon-670K``
        - 0.55: otherwise

    :type A: float, optional
    :param B: B value, typical values:

        - 0.4: ``WikiLSHTC-325K``
        - 2.6: ``Amazon-670K``
        - 1.5: otherwise

    :type B: float, optional
    :return: ndarray with propensity scores for each label
    """
    if isinstance(Y, np.ndarray) or isinstance(Y, csr_matrix):
        m = Y.shape[0]
        freqs = np.sum(Y, axis=0)

    elif all((isinstance(y, list) or isinstance(y, tuple)) for y in Y):
        m = max([max(y) for y in Y])
        freqs = np.zeros(m)
        for y in Y:
            freqs[y] += 1

    else:
        raise TypeError("Unsupported data type, should be Numpy matrix, Scipy sparse matrix or list of list of ints")

    C = (log(m) - 1.0) * (B + 1) ** A
    inv_ps = 1.0 + C * (freqs + B) ** -A
    return inv_ps


def psprecision_at_k(Y_true, Y_pred, inv_ps, k=5):
    """
    Calculate Propensity Scored Precision (PSP) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels
    :type Y_true: ndarray, csr_matrix, list[list[int]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as list of labels or tuples of labels with scores (idx, score)..
        In case of matrix, ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
    :param inv_ps: Propensity scores for each label.
    :type inv_ps: ndarray, list
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: ndarray with values of PSP at 1-k places.
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    if not isinstance(inv_ps, np.ndarray):
        inv_ps = np.array(inv_ps)

    sum = np.zeros(k)
    best_sum = np.zeros(k)
    for t, p in zip(Y_true, Y_pred):
        top_ps = np.sort(inv_ps[t])
        psp_at_i = 0
        best_psp_at_i = 0
        for i, p_i in enumerate(p[:k]):
            psp_at_i += inv_ps[p_i] if p_i in t else 0
            if i < top_ps.shape:
                best_psp_at_i += top_ps[i]
            sum[i] += psp_at_i / (i + 1)
            best_sum[i] += best_psp_at_i / (i + 1)
    return sum / best_sum


# Helpers
def _check_k(k):
    if not isinstance(k, int):
        raise TypeError("k should be an integer number larger than 0")
    if k < 1:
        raise ValueError("k should be larger than 0")


def _Y_np_iterator(Y, ranking=False):
    if ranking:
        for i in range(0, Y.shape[0]):
            yield (-Y[i]).argsort()
    else:
        for i in range(0, Y.shape[0]):
            yield Y[i].nonzero()[0]


def _Y_csr_matrix_iterator(Y, ranking=False):
    if ranking:
        for i in range(0, Y.shape[0]):
            ranking = (-Y[i].data).argsort()
            yield Y[i].indices[ranking]
    else:
        for i in range(0, Y.shape[0]):
            yield Y[i].indices


def _Y_list_iterator(Y):
    for y in Y:
        if all(isinstance(y_i, tuple) for y_i in y):
            #yield [y_i[0] for y_i in sorted(y, key=lambda y_i: y_i[1], reverse=True)]
            yield [y_i[0] for y_i in y]
        else:
            yield y


def _get_Y_iterator(Y, ranking=False):
    if isinstance(Y, np.ndarray):
        return _Y_np_iterator(Y, ranking)

    elif isinstance(Y, csr_matrix):
        return _Y_csr_matrix_iterator(Y, ranking)

    elif all((isinstance(y, list) or isinstance(y, tuple)) for y in Y):
        return _Y_list_iterator(Y)

    else:
        raise TypeError("Unsupported data type, should be Numpy matrix, Scipy sparse matrix or list of list of ints")
