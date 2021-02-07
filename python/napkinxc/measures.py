# Copyright (c) 2020-2021 by Marek Wydmuch
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

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: Values of precision at 1-k places.
    :rtype: ndarray
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

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param zero_division: Value to add when there is a zero division, typically set to 0, defaults to 0
    :type zero_division: float, optional
    :return: Values of recall at 1-k places.
    :rtype: ndarray
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

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: Values of coverage at 1-k places.
    :rtype: ndarray
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    uniq_t = set()
    uniq_tp_at_i = [set() for _ in range(k)]
    for t, p in zip(Y_true, Y_pred):
        for t_i in t:
            uniq_t.add(t_i)
        for i, p_i in enumerate(p[:k]):
            if p_i in t:
                uniq_tp_at_i[i].add(p_i)
    uniq_tp = np.zeros(k)
    for i in range(0, k):
        if i > 0:
            uniq_tp_at_i[i].update(uniq_tp_at_i[i - 1])
        uniq_tp[i] = len(uniq_tp_at_i[i]) / len(uniq_t)

    return uniq_tp


def dcg_at_k(Y_true, Y_pred, k=5):
    """
    Calculate Discounted Cumulative Gain (DCG) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: Values of DCG at 1-k places.
    :rtype: ndarray
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


def ndcg_at_k(Y_true, Y_pred, k=5, zero_division=0):
    """
    Calculate normalized Discounted Cumulative Gain (nDCG) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param zero_division: Value to add when there is a zero division, typically set to 0, defaults to 0
    :type zero_division: float, optional
    :return: Values of nDCG at 1-k places.
    :rtype: ndarray
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)

    sum = np.zeros(k)
    count = 0
    for t, p in zip(Y_true, Y_pred):
        count += 1
        dcg_at_i = 0
        norm_at_i = 0
        norm_len = min(k, len(t))
        if norm_len == 0:
            sum += zero_division
            continue
        for i, p_i in enumerate(p[:k]):
            _log_i = 1 / log2(i + 2)
            if i < norm_len:
                norm_at_i += _log_i
            dcg_at_i += 1 * _log_i if p_i in t else 0
            sum[i] += dcg_at_i / norm_at_i
    return sum / count


def hamming_loss(Y_true, Y_pred):
    """
    Calculate unnormalized (to avoid very small numbers because of large number of labels) hamming loss - average number of misclassified labels.
    
    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred: Predicted labels provided as a matrix with scores or list of lists of labels or tuples of labels with scores (label, score).
    :type Y_pred: ndarray, csr_matrix, list[list|set[int|str]], list[list|set[tuple[int|str, float]]
    :return: Value of hamming loss.
    :rtype: float
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

    :param Y: Labels (typically ground truth for train data) provided as a matrix with non-zero values for relevant labels.
    :type Y: ndarray, csr_matrix, list[list[tuple[int|str, float]]
    :param A: Dataset specific parameter, typical values:

        - 0.5: ``WikiLSHTC-325K`` and ``WikipediaLarge-500K``
        - 0.6: ``Amazon-670K`` and ``Amazon-3M``
        - 0.55: otherwise

        Defaults to 0.55
    :type A: float, optional
    :param B: Dataset specific parameter, typical values:

        - 0.4: ``WikiLSHTC-325K`` and ``WikipediaLarge-500K``
        - 2.6: ``Amazon-670K`` and ``Amazon-3M``
        - 1.5: otherwise

        Defaults to 1.5
    :type B: float, optional
    :return: Array with the inverse propensity for all labels
    :rtype: ndarray
    """
    if isinstance(Y, np.ndarray) or isinstance(Y, csr_matrix):
        n, m = Y.shape
        freqs = np.ravel(np.sum(Y, axis=0))

    elif all((isinstance(y, list) or isinstance(y, tuple)) for y in Y):
        n = len(Y)
        m = max([max(y) for y in Y if len(y)])
        freqs = np.zeros(m + 1)
        for y in Y:
            freqs[y] += 1

    else:
        raise TypeError("Unsupported data type, should be Numpy matrix, Scipy sparse matrix or list of list of ints")

    C = (log(n) - 1) * (B + 1) ** A
    inv_ps = 1 + C * (freqs + B) ** -A
    return inv_ps


def psprecision_at_k(Y_true, Y_pred, inv_ps, k=5):
    """
    Calculate Propensity Scored Precision (PSP) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Propensity scores for each label. In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: Values of PSP at 1-k places.
    :rtype: ndarray
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)
    inv_ps, _top_ps = _get_top_ps_func(inv_ps)

    sum = np.zeros(k)
    best_sum = np.zeros(k)
    for t, p in zip(Y_true, Y_pred):
        top_ps = _top_ps(inv_ps, t)
        psp_at_i = 0
        best_psp_at_i = 0
        for i, p_i in enumerate(p[:k]):
            psp_at_i += inv_ps[p_i] if p_i in t else 0
            if i < top_ps.shape[0]:
                best_psp_at_i += top_ps[i]
            sum[i] += psp_at_i / (i + 1)
            best_sum[i] += best_psp_at_i / (i + 1)
    return sum / best_sum


def psrecall_at_k(Y_true, Y_pred, inv_ps, k=5, zero_division=0):
    """
    Calculate Propensity Scored Recall (PSR) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Propensity scores for each label. In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param zero_division: Value to add when there is a zero division, typically set to 0, defaults to 0
    :type zero_division: float, optional
    :return: Values of PSR at 1-k places.
    :rtype: ndarray
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)
    inv_ps, _top_ps = _get_top_ps_func(inv_ps)

    sum = np.zeros(k)
    best_sum = np.zeros(k)
    for t, p in zip(Y_true, Y_pred):
        if len(t) == 0:
            sum += zero_division
            best_sum += zero_division
            continue
        psr_at_i = 0
        best_psr_at_i = 0
        top_ps = _top_ps(inv_ps, t)
        for i, p_i in enumerate(p[:k]):
            psr_at_i += inv_ps[p_i] if p_i in t else 0
            if i < top_ps.shape[0]:
                best_psr_at_i += top_ps[i]
            sum[i] += psr_at_i / len(t)
            best_sum[i] += best_psr_at_i / len(t)
    return sum / best_sum


def psdcg_at_k(Y_true, Y_pred, inv_ps, k=5):
    """
    Calculate Propensity Scored Discounted Cumulative Gain (PSDCG) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Propensity scores for each label. In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :return: Values of PSDCG at 1-k places.
    :rtype: ndarray
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)
    inv_ps, _top_ps = _get_top_ps_func(inv_ps)

    sum = np.zeros(k)
    best_sum = np.zeros(k)
    for t, p in zip(Y_true, Y_pred):
        psdcg_at_i = 0
        best_psdcg_at_i = 0
        top_ps = _top_ps(inv_ps, t)
        for i, p_i in enumerate(p[:k]):
            _log_i = 1 / log2(i + 2)
            psdcg_at_i += inv_ps[p_i] * _log_i if p_i in t else 0
            if i < top_ps.shape[0]:
                best_psdcg_at_i += top_ps[i] * _log_i
            sum[i] += psdcg_at_i / (i + 1)
            best_sum[i] += best_psdcg_at_i / (i + 1)
    return sum / best_sum


def psndcg_at_k(Y_true, Y_pred, inv_ps, k=5, zero_division=0):
    """
    Calculate Propensity Scored normalized Discounted Cumulative Gain (PSnDCG) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Propensity scores for each label. In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param zero_division: Value to add when there is a zero division, typically set to 0, defaults to 0
    :type zero_division: float, optional
    :return: Values of PSnDCG at 1-k places.
    :rtype: ndarray
    """
    _check_k(k)
    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred, ranking=True)
    inv_ps, _top_ps = _get_top_ps_func(inv_ps)

    sum = np.zeros(k)
    best_sum = np.zeros(k)
    for t, p in zip(Y_true, Y_pred):
        psdcg_at_i = 0
        best_psdcg_at_i = 0
        norm_at_i = 0
        norm_len = min(k, len(t))
        if norm_len == 0:
            sum += zero_division
            best_sum += zero_division
            continue

        top_ps = _top_ps(inv_ps, t)
        for i, p_i in enumerate(p[:k]):
            _log_i = 1 / log2(i + 2)
            if i < norm_len:
                norm_at_i += _log_i
            psdcg_at_i += inv_ps[p_i] * _log_i if p_i in t else 0
            if i < top_ps.shape[0]:
                best_psdcg_at_i += top_ps[i] * _log_i
            sum[i] += psdcg_at_i / norm_at_i
            best_sum[i] += best_psdcg_at_i / norm_at_i
    return sum / best_sum


def f1_measure(Y_true, Y_pred, average='micro', zero_division=0):
    """
    Calculate F1 measure, also known as balanced F-score or F-measure.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred: Predicted labels provided as a matrix with scores or list of lists of labels or tuples of labels with scores (label, score).
    :type Y_pred: ndarray, csr_matrix, list[list|set[int|str]], list[list|set[tuple[int|str, float]]
    :param average: Determines the type of performed averaging {``'micro'``, ``'macro'``, ``'sample'``}, default to ``'micro'``
    :type average: str
    :param zero_division: Value to add when there is a zero division, typically set to 0, defaults to 0
    :type zero_division: float, optional
    :return: Value of F1-measure.
    :rtype: float
    """

    Y_true = _get_Y_iterator(Y_true)
    Y_pred = _get_Y_iterator(Y_pred)

    sum = 0
    count = 0
    if average == 'micro':
        for t, p in zip(Y_true, Y_pred):
            tp = len(set(t).intersection(p))
            fp = len(p) - tp
            fn = len(t) - tp
            sum += 2 * tp
            count += 2 * tp + fp + fn

    elif average == 'macro':
        labels_tp = {}
        labels_fp = {}
        labels_fn = {}
        for t, p in zip(Y_true, Y_pred):
            tp = set(t).intersection(p)
            for tp_i in tp:
                labels_tp[tp_i] = labels_tp.get(tp_i, 0) + 1
            for p_i in p:
                if p_i not in tp:
                    labels_fp[p_i] = labels_fp.get(p_i, 0) + 1
            for t_i in t:
                if t_i not in tp:
                    labels_fn[t_i] = labels_fn.get(t_i, 0) + 1

        labels = set(list(labels_tp.keys()) + list(labels_fp.keys()) + list(labels_fn.keys()))
        if all(isinstance(l, (int, np.integer)) for l in labels): # If there is no text labels
            max_label = max(max(labels_tp.keys()), max(labels_fp.keys()), max(labels_fn.keys()))
            labels = range(max_label + 1)

        for l in labels:
            if (2 * labels_tp.get(l, 0) + labels_fp.get(l, 0) + labels_fn.get(l, 0)) > 0:
                sum += 2 * labels_tp.get(l, 0) / (2 * labels_tp.get(l, 0) + labels_fp.get(l, 0) + labels_fn.get(l, 0))
            else:
                sum += zero_division
            count += 1

    elif average == 'samples':
        for t, p in zip(Y_true, Y_pred):
            tp = len(set(t).intersection(p))
            precision = tp / len(p) if len(p) > 0 else 0
            recall = tp / len(t) if len(t) > 0 else zero_division
            if recall > 0:
                sum += 2 * (precision * recall) / (precision + recall)
            else:
                sum += zero_division
            count += 1

    else:
        raise ValueError("average should be in {'micro', 'macro', 'samples'}")

    return sum / count


# Helpers
def _check_k(k):
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k should be an integer number larger than 0")
    if k < 1:
        raise ValueError("k should be larger than 0")


def _Y_np_iterator(Y, ranking=False):
    rows = Y.shape[0]
    if ranking:
        for i in range(0, rows):
            yield (-Y[i]).argsort()
    else:
        for i in range(0, rows):
            yield Y[i].nonzero()[0]


def _Y_csr_matrix_iterator(Y, ranking=False):
    rows = Y.shape[0]
    if ranking:
        for i in range(0, rows):
            y = Y[i]
            ranking = (-y.data).argsort()
            yield y.indices[ranking]
    else:
        for i in range(0, rows):
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

    elif all(isinstance(y, (list, tuple)) for y in Y):
        return _Y_list_iterator(Y)

    else:
        raise TypeError("Unsupported data type, should be Numpy matrix (2d array), or Scipy CSR matrix, or list of list of ints")


def _top_ps_dict(inv_ps, t):
    return np.array(sorted([inv_ps.get(t_i, 0) for t_i in t], reverse=True))


def _top_ps_np(inv_ps, t):
    t = [t_i for t_i in t if t_i < inv_ps.shape[0]]
    return -np.sort(-inv_ps[t])


def _get_top_ps_func(inv_ps):
    if isinstance(inv_ps, dict):
        _top_ps = _top_ps_dict
    elif isinstance(inv_ps, list):
        inv_ps = np.array(inv_ps)
        _top_ps = _top_ps_np
    elif isinstance(inv_ps, np.ndarray):
        _top_ps = _top_ps_np
    else:
        raise TypeError("Unsupported data type for inv_ps, should be Numpy vector (1d array), or list, or dict")

    return inv_ps, _top_ps
