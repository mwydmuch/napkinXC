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


def precision_at_k(Y_true, Y_pred, k=5):
    """
    Calculate precision at {1-k} places

    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of precision at k
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
    Calculate recall at k

    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :param zero_division: sets the value to use when there is a zero division for an instance caused by number of true labels equal to 0, should be 0 or 1
    :return: value of recall at k
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
    Calculate coverage at k

    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of coverage at k
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    uniq_t = set()
    uniq_tp = set()
    for t, p in zip(Y_true, Y_pred):
        for t_i in t:
            uniq_t.add(t_i)
        for p_i in p[:k]:
            if p_i in t:
                uniq_tp.add(p_i)
    return len(uniq_tp) / len(uniq_t)


def dcg_at_k(Y_true, Y_pred, k=5):
    """
    Calculate DCG at k

    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of DCG at k
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
    Calculate nDCG at k

    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of nDCG at k
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
    Calculate hamming loss

    :param Y_true: true labels
    :param Y_pred: predicted labels
    :return: value of hamming loss
    """
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred)

    sum = 0
    count = 0
    for t, p in zip(Y_true, Y_pred):
        sum += len(p) + len(t) - 2 * len(set(t).intersection(p))
        count += 1

    return sum / count


def inverse_propensity(Y, A, B):
    """
    Computes inverse propernsity as proposed in Jain et al. 16.

    :param Y:
    :param A:
    :param B:
    :return:
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
    inv_psp = 1.0 + C * (freqs + B) ** -A
    return inv_psp


def psprecision_at_k(Y_true, Y_pred, inv_psp, k=5):
    """
    Calculate precision at {1-k} places

    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of precision at k
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    if not isinstance(inv_psp, np.ndarray):
        inv_psp = np.array(inv_psp)

    sum = np.zeros(k)
    best_sum = np.zeros(k)
    for t, p in zip(Y_true, Y_pred):
        top_psp = np.sort(inv_psp[t])
        psp_at_i = 0
        best_psp_at_i = 0
        for i, p_i in enumerate(p[:k]):
            psp_at_i += inv_psp[p_i] if p_i in t else 0
            if i < top_psp.shape:
                best_psp_at_i += best_psp[i]
            sum[i] += psp_at_i / (i + 1)
            best_sum[i] += best_psp_at_i / (i + 1)
    return sum / best_sum
