# Copyright (c) 2020-2025 by Marek Wydmuch
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

from abc import ABC, abstractmethod
from math import log2, log
import numpy as np
from scipy.sparse import csr_matrix
from collections.abc import Iterable


# TODOs:
# - Add docstrings to classes
# - Add macro measures at k?
# - Add measure dict class
# - Add F-beta variant of measure?
# - Add normalization to hamming loss?


# Classes for different measures

class Metric(ABC):
    """
    Abstract class for measure.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.needs_ranking = False
        self.sum = 0
        self.count = 0

    def accumulate(self, Y_true, Y_pred):
        Y_true = Metric._get_Y_iterator(Y_true)
        Y_pred = Metric._get_Y_iterator(Y_pred, ranking=self.needs_ranking)

        for t, p in zip(Y_true, Y_pred):
            self._accumulate(t, p)
            self.count += 1

    def __call__(self, Y_true, Y_pred):
        self.accumulate(Y_true, Y_pred)

    @abstractmethod
    def _accumulate(self, t, p):
        raise NotImplementedError

    def summarize(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def calculate(self, Y_true, Y_pred):
        self.reset()
        self.accumulate(Y_true, Y_pred)
        return self.summarize()

    @staticmethod
    def _get_Y_iterator(Y, ranking=False):
        if isinstance(Y, np.ndarray):
            return Metric._Y_np_iterator(Y, ranking=ranking)

        elif isinstance(Y, csr_matrix):
            return Metric._Y_csr_matrix_iterator(Y, ranking=ranking)

        elif all(isinstance(y, (list, tuple)) for y in Y):
            return Metric._Y_list_iterator(Y)

        else:
            raise TypeError("Unsupported data type, should be Numpy matrix (2d array), or Scipy CSR matrix, or list of list of ints")

    @staticmethod
    def _Y_np_iterator(Y, ranking=False):
        rows = Y.shape[0]
        if ranking:
            for i in range(0, rows):
                yield (-Y[i]).argsort()
        else:
            for i in range(0, rows):
                yield Y[i].nonzero()[0]

    @staticmethod
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

    @staticmethod
    def _Y_list_iterator(Y):
        for y in Y:
            if all(isinstance(y_i, tuple) for y_i in y):
                #yield [y_i[0] for y_i in sorted(y, key=lambda y_i: y_i[1], reverse=True)]
                yield [y_i[0] for y_i in y]
            else:
                yield y


class MetricAtK(Metric):
    """
    Abstract class for measure calculated at 1-k places.
    """
    def __init__(self, k=5, **kwargs):
        super().__init__(**kwargs)
        MetricAtK._check_k(k)
        self.k = k
        self.sum = np.zeros(self.k)
        self.needs_ranking = True

    def accumulate(self, Y_true, Y_pred):
        Y_true = Metric._get_Y_iterator(Y_true)
        Y_pred = Metric._get_Y_iterator(Y_pred, ranking=True)

        for t, p in zip(Y_true, Y_pred):
            self._accumulate(t, p)
            self.count += 1

    def reset(self):
        self.sum = np.zeros(self.k)
        self.count = 0

    @staticmethod
    def _check_k(k):
        if not isinstance(k, (int, np.integer)):
            raise TypeError("k should be an integer number larger than 0")
        if k < 1:
            raise ValueError("k should be larger than 0")


class PSMetricAtK(MetricAtK):
    """
    Abstract class for Propensity Scored measure calculated at 1-k places.
    """
    def __init__(self, inv_ps, k=5, normalize=True, **kwargs):
        super().__init__(k=k, **kwargs)
        self.inv_ps, self._top_ps = PSMetricAtK._get_top_ps_func(inv_ps)
        self.best_sum = np.zeros(self.k)
        self.normalize = normalize

    def summarize(self):
        return self.sum / (self.best_sum if self.normalize else self.count)

    @staticmethod
    def _top_ps_dict(inv_ps, t):
        return np.array(sorted([inv_ps.get(t_i, 0) for t_i in t], reverse=True))

    @staticmethod
    def _top_ps_np(inv_ps, t):
        t = [t_i for t_i in t if t_i < inv_ps.shape[0]]
        return -np.sort(-inv_ps[t])

    @staticmethod
    def _get_top_ps_func(inv_ps):
        if isinstance(inv_ps, dict):
            _top_ps = PSMetricAtK._top_ps_dict
        elif isinstance(inv_ps, list):
            inv_ps = np.array(inv_ps)
            _top_ps = PSMetricAtK._top_ps_np
        elif isinstance(inv_ps, np.ndarray):
            _top_ps = PSMetricAtK._top_ps_np
        else:
            raise TypeError("Unsupported data type for inv_ps, should be Numpy vector (1d array), or list, or dict")

        return inv_ps, _top_ps



# Popular standard measures

class HammingLoss(Metric):
    def __init__(self):
        super().__init__()

    def _accumulate(self, t, p):
        self.sum += len(p) + len(t) - 2 * len(set(t).intersection(p))


class PrecisionAtK(MetricAtK):
    def __init__(self, k=5):
        super().__init__(k=k)

    def _accumulate(self, t, p):
        p_at_i = 0
        for i in range(self.k):
            if i < len(p):
                p_at_i += 1 if p[i] in t else 0
            self.sum[i] += p_at_i / (i + 1)


class RecallAtK(MetricAtK):
    def __init__(self, k=5, zero_division=0):
        super().__init__(k=k)
        self.zero_division = zero_division

    def _accumulate(self, t, p):
        if len(t) > 0:
            r_at_i = 0
            for i in range(self.k):
                if i < len(p):
                    r_at_i += 1 if p[i] in t else 0
                self.sum[i] += r_at_i / len(t)
        else:
            self.sum += self.zero_division


class DCGAtK(MetricAtK):
    def __init__(self, k=5):
        super().__init__(k=k)

    def _accumulate(self, t, p):
        dcg_at_i = 0
        for i in range(self.k):
            if i < len(p):
                dcg_at_i += 1 / log2(i + 2) if p[i] in t else 0
            self.sum[i] += dcg_at_i


class NDCGAtK(MetricAtK):
    def __init__(self, k=5, zero_division=0):
        super().__init__(k=k)
        self.zero_division = zero_division

    def _accumulate(self, t, p):
        dcg_at_i = 0
        norm_at_i = 0
        norm_len = min(self.k, len(t))
        if norm_len == 0:
            self.sum += self.zero_division
        else:
            for i in range(self.k):
                _log_i = 1 / log2(i + 2)
                if i < norm_len:
                    norm_at_i += _log_i
                if i < len(p):
                    dcg_at_i += 1 * _log_i if p[i] in t else 0
                self.sum[i] += dcg_at_i / norm_at_i


# Propensity scored (weighted) measures (unbiased variants of standard measures)

class PSPrecisionAtK(PSMetricAtK):
    def __init__(self, inv_ps, k=5, normalize=True):
        super().__init__(inv_ps, k=k, normalize=normalize)

    def _accumulate(self, t, p):
        psp_at_i = 0
        best_psp_at_i = 0
        top_ps = self._top_ps(self.inv_ps, t)
        for i in range(self.k):
            if i < len(p):
                psp_at_i += self.inv_ps[p[i]] if p[i] in t else 0
            if i < top_ps.shape[0]:
                best_psp_at_i += top_ps[i]
            self.sum[i] += psp_at_i / (i + 1)
            self.best_sum[i] += best_psp_at_i / (i + 1)


class PSRecallAtK(PSMetricAtK):
    def __init__(self, inv_ps, k=5, normalize=True, zero_division=0):
        super().__init__(inv_ps, k=k, normalize=normalize)
        self.zero_division = zero_division

    def _accumulate(self, t, p):
        if len(t) == 0:
            self.sum += self.zero_division
            self.best_sum += self.zero_division
        else:
            psr_at_i = 0
            best_psr_at_i = 0
            top_ps = self._top_ps(self.inv_ps, t)
            for i in range(self.k):
                if i < len(p):
                    psr_at_i += self.inv_ps[p[i]] if p[i] in t else 0
                if i < top_ps.shape[0]:
                    best_psr_at_i += top_ps[i]
                self.sum[i] += psr_at_i / len(t)
                self.best_sum[i] += best_psr_at_i / len(t)


class PSDCGAtK(PSMetricAtK):
    def __init__(self, inv_ps, k=5, normalize=True):
        super().__init__(inv_ps, k=k, normalize=normalize)

    def _accumulate(self, t, p):
        psdcg_at_i = 0
        best_psdcg_at_i = 0
        top_ps = self._top_ps(self.inv_ps, t)
        for i in range(self.k):
            _log_i = 1 / log2(i + 2)
            if i < len(p):
                psdcg_at_i += self.inv_ps[p[i]] * _log_i if p[i] in t else 0
            if i < top_ps.shape[0]:
                best_psdcg_at_i += top_ps[i] * _log_i
            self. sum[i] += psdcg_at_i / (i + 1)
            self.best_sum[i] += best_psdcg_at_i / (i + 1)


class PSNDCGAtK(PSMetricAtK):
    def __init__(self, inv_ps, k=5, normalize=True, zero_division=0):
        super().__init__(inv_ps, k=k, normalize=normalize)
        self.zero_division = zero_division

    def _accumulate(self, t, p):
        psdcg_at_i = 0
        best_psdcg_at_i = 0
        norm_at_i = 0
        norm_len = min(self.k, len(t))
        if norm_len == 0:
            self.sum += self.zero_division
            self.best_sum += self.zero_division
        else:
            top_ps = self._top_ps(self.inv_ps, t)
            for i in range(self.k):
                _log_i = 1 / log2(i + 2)
                if i < norm_len:
                    norm_at_i += _log_i
                if i < len(p):
                    psdcg_at_i += self.inv_ps[p[i]] * _log_i if p[i] in t else 0
                if i < top_ps.shape[0]:
                    best_psdcg_at_i += top_ps[i] * _log_i
                self.sum[i] += psdcg_at_i / norm_at_i
                self.best_sum[i] += best_psdcg_at_i / norm_at_i


# Other measures

class AbandonmentAtK(MetricAtK):
    def __init__(self, k=5):
        super().__init__(k=k)

    def _accumulate(self, t, p):
        a_at_i = 0
        for i in range(self.k):
            if i < len(p):
                a_at_i = 1 if p[i] in t else a_at_i
            self.sum[i] += a_at_i


class CoverageAtK(MetricAtK):
    def __init__(self, k=5):
        super().__init__(k=k)
        self.uniq_t = set()
        self.uniq_tp_at_i = [set() for _ in range(self.k)]

    def _accumulate(self, t, p):
        for t_i in t:
            self.uniq_t.add(t_i)
        for i, p_i in enumerate(p[:self.k]):
            if p_i in t:
                for j in range(i, self.k):
                    self.uniq_tp_at_i[j].add(p_i)

    def summarize(self):
        uniq_tp = np.zeros(self.k)
        for i in range(0, self.k):
            uniq_tp[i] = len(self.uniq_tp_at_i[i]) / len(self.uniq_t)

        return uniq_tp

    def reset(self):
        super().reset()
        self.uniq_t = set()
        self.uniq_tp_at_i = [set() for _ in range(self.k)]


class MicroF1Metric(Metric):
    def __init__(self):
        super().__init__()

    def _accumulate(self, t, p):
        tp = len(set(t).intersection(p))
        fp = len(p) - tp
        fn = len(t) - tp
        self.sum += 2 * tp
        self.count += 2 * tp + fp + fn
        self.count -= 1  # Because of count += 1 in accumulate


class SamplesF1Metric(Metric):
    def __init__(self, zero_division=0):
        super().__init__()
        self.zero_division = zero_division
        
    def _accumulate(self, t, p):
        tp = len(set(t).intersection(p))
        precision = tp / len(p) if len(p) > 0 else 0
        recall = tp / len(t) if len(t) > 0 else self.zero_division
        if recall > 0:
            self.sum += 2 * (precision * recall) / (precision + recall)
        else:
            self.sum += self.zero_division


# Macro measures


class MacroMetric(Metric):
    """
    Abstract class for macro measures.
    """
    def __init__(self, zero_division=0, **kwargs):
        super().__init__(**kwargs)
        self.zero_division = zero_division
        self.labels_tp = {}
        self.labels_fp = {}
        self.labels_fn = {}

    def reset(self):
        super().reset()
        self.labels_tp = {}
        self.labels_fp = {}
        self.labels_fn = {}

    def _accumulate_conf_matrix(self, t, p, tp, labels_tp, labels_fp, labels_fn):
        for tp_i in tp:
            labels_tp[tp_i] = labels_tp.get(tp_i, 0) + 1
        for p_i in p:
            if p_i not in tp:
                labels_fp[p_i] = labels_fp.get(p_i, 0) + 1
        for t_i in t:
            if t_i not in tp:
                labels_fn[t_i] = labels_fn.get(t_i, 0) + 1

    def _accumulate(self, t, p):
        tp = set(t).intersection(p)
        self._accumulate_conf_matrix(t, p, tp, self.labels_tp, self.labels_fp, self.labels_fn)

    @abstractmethod
    def _summarize_conf_matrix(self, l_tp, l_fp, l_fn):
        raise NotImplementedError

    def _summarize(self, labels_tp, labels_fp, labels_fn):
        labels = set(list(labels_tp.keys()) + list(labels_fp.keys()) + list(labels_fn.keys()))
        if all(isinstance(l, (int, np.integer)) for l in labels):  # If there is no text labels
            max_label = max(max(labels_tp.keys()), max(labels_fp.keys()), max(labels_fn.keys()))
            labels = range(max_label + 1)

        sum = 0
        denominator = 0
        for l in labels:
            l_tp = labels_tp.get(l, 0)
            l_fp = labels_fp.get(l, 0)
            l_fn = labels_fn.get(l, 0)

            if self._check_div_zero(l_tp, l_fp, l_fn):
                sum += self._summarize_conf_matrix(l_tp, l_fp, l_fn)
            else:
                sum += self.zero_division
            denominator += 1

        return sum / denominator

    def summarize(self):
        return self._summarize(self.labels_tp, self.labels_fp, self.labels_fn)


class MacroF1Metric(MacroMetric):
    def __init__(self, zero_division=0):
        super().__init__(zero_division=zero_division)

    def _check_div_zero(self, l_tp, l_fp, l_fn):
        return (l_tp + l_fp + l_fn) > 0

    def _summarize_conf_matrix(self, l_tp, l_fp, l_fn):
        return 2 * l_tp / (2 * l_tp + l_fp + l_fn)


class MacroMetricAtK(MetricAtK, MacroMetric):
    """
    Abstract class for macro measures calculated at k-place.
    """
    def __init__(self, k=5, zero_division=0, **kwargs):
        super().__init__(k=5, zero_division=zero_division, **kwargs)
        self.labels_tp = [{} for _ in range(self.k)]
        self.labels_fp = [{} for _ in range(self.k)]
        self.labels_fn = [{} for _ in range(self.k)]

    def reset(self):
        super().reset()
        self.labels_tp = [{} for _ in range(self.k)]
        self.labels_fp = [{} for _ in range(self.k)]
        self.labels_fn = [{} for _ in range(self.k)]

    def _accumulate(self, t, p):
        tp_at_k = set()
        for i in range(self.k):
            if i < len(p) and p[i] in t:
                tp_at_k.add(p[i])
            self._accumulate_conf_matrix(t, p[:i+1], tp_at_k, self.labels_tp[i], self.labels_fp[i], self.labels_fn[i])

    def summarize(self):
        results = np.zeros(self.k)
        for i in range(self.k):
            results[i] = self._summarize(self.labels_tp[i], self.labels_fp[i], self.labels_fn[i])
        return results


class MacroPrecisionAtK(MacroMetricAtK):
    def __init__(self, k=5, zero_division=0):
        super().__init__(k=k, zero_division=zero_division)

    def _check_div_zero(self, l_tp, l_fp, l_fn):
        return (l_tp + l_fp) > 0

    def _summarize_conf_matrix(self, l_tp, l_fp, l_fn):
        return l_tp / (l_tp + l_fp)


class MacroRecallAtK(MacroMetricAtK):
    def __init__(self, k=5, zero_division=0):
        super().__init__(k=k, zero_division=zero_division)

    def _check_div_zero(self, l_tp, l_fp, l_fn):
        return (l_tp + l_fn) > 0

    def _summarize_conf_matrix(self, l_tp, l_fp, l_fn):
        return l_tp / (l_tp + l_fn)


class MacroF1MetricAtK(MacroMetricAtK):
    def __init__(self, k=5, zero_division=0):
        super().__init__(k=k, zero_division=zero_division)

    def _check_div_zero(self, l_tp, l_fp, l_fn):
        return (l_tp + l_fp + l_fn) > 0

    def _summarize_conf_matrix(self, l_tp, l_fp, l_fn):
        return 2 * l_tp / (2 * l_tp + l_fp + l_fn)




# Functions for different measures


def precision_at_k(Y_true, Y_pred, k=5):
    """
    Calculate precision at 1-k places.
    Precision at k is defined as:

    .. math::

        p@k = \\frac{1}{k} \\sum_{l \\in \\text{rank}_k(\\hat{\\pmb{y}})} y_l \\,,

    where :math:`\\pmb{y} \\in {0, 1}^m` is ground truth label vector,
    :math:`\\hat{\\pmb{y}} \\in \\mathbb{R}^m` is predicted labels score vector,
    and :math:`\\text{rank}_k(\\hat{\\pmb{y}})` returns the :math:`k` indices of :math:`\\hat{\\pmb{y}}`
    with the largest values, ordered in descending order.

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
    return PrecisionAtK(k=k).calculate(Y_true, Y_pred)


def recall_at_k(Y_true, Y_pred, k=5, zero_division=0):
    """
    Calculate recall at 1-k places.
    Recall at k is defined as:

    .. math::

        r@k = \\frac{1}{||\\pmb{y}||_1} \\sum_{l \\in \\text{rank}_k(\\hat{\\pmb{y}})} y_l \\,,

    where :math:`\\pmb{y} \\in {0, 1}^m` is ground truth label vector,
    :math:`\\hat{\\pmb{y}} \\in \\mathbb{R}^m` is predicted labels score vector,
    and :math:`\\text{rank}_k(\\hat{\\pmb{y}})` returns the :math:`k` indices of :math:`\\hat{\\pmb{y}}`
    with the largest values, ordered in descending order.

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
    return RecallAtK(k=k, zero_division=zero_division).calculate(Y_true, Y_pred)


def abandonment_at_k(Y_true, Y_pred, k=5):
    """
    Calculate abandonment at 1-k places.

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
    return AbandonmentAtK(k=k).calculate(Y_true, Y_pred)


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
    return CoverageAtK(k=k).calculate(Y_true, Y_pred)


def dcg_at_k(Y_true, Y_pred, k=5):
    """
    Calculate Discounted Cumulative Gain (DCG) at 1-k places.
    DCG at k is defined as:

    .. math::

        DCG@k = \\sum_{i = 1}^{k} \\frac{y_{\\text{rank}_k(\\hat{\\pmb{y}})_i}}{\\log_2(i + 1)} \\,,

    where :math:`\\pmb{y} \\in {0, 1}^m` is ground truth label vector,
    :math:`\\hat{\\pmb{y}} \\in \\mathbb{R}^m` is predicted labels score vector,
    and :math:`\\text{rank}_k(\\hat{\\pmb{y}})` returns the :math:`k` indices of :math:`\\hat{\\pmb{y}}`
    with the largest values, ordered in descending order.

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
    return DCGAtK(k=k).calculate(Y_true, Y_pred)


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
    return NDCGAtK(k=k, zero_division=zero_division).calculate(Y_true, Y_pred)


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
    return HammingLoss().calculate(Y_true, Y_pred)


def count_labels(Y):
    """
    Count number of occurrences of each label.

    :param Y: Labels (typically ground truth for train data) provided as a matrix with non-zero values for relevant labels.
    :type Y: ndarray, csr_matrix, list[list[int|str]]
    :return: Array with the count of labels occurrences.
    :rtype: ndarray
    """
    if isinstance(Y, (np.ndarray, csr_matrix)):
        counts = np.ravel(np.sum(Y, axis=0))

    elif isinstance(Y, Iterable) and all(isinstance(y, (list, tuple)) for y in Y):
        m = max([max(y) for y in Y if len(y)])
        
        if isinstance(m, int):    
            counts = np.zeros(m + 1)
            for y in Y:
                counts[y] += 1
        elif isinstance(m, str):
            counts = {}
            for y in Y:
                counts[y] = counts.get(y, 0) + 1
        else:
            raise TypeError("Unsupported data type, labels should be integers or strings")

    else:
        raise TypeError(
            "Unsupported data type, should be Numpy matrix (2d array), Scipy sparse matrix or list of lists of ints")

    return counts


def labels_priors(Y):
    """
    Calculate prior probablity of each label.

    :param Y: Labels (typically ground truth for train data) provided as a matrix with non-zero values for relevant labels.
    :type Y: ndarray, csr_matrix, list[list[int]]
    :return: Array with the prior probabilities of labels.
    :rtype: ndarray
    """
    counts = count_labels(Y)
    if isinstance(Y, (np.ndarray, csr_matrix)):
        return counts / Y.shape[0]

    else:
        return counts / len(Y)


def inverse_labels_priors(Y):
    """
    Calculate inverse of prior probablity of each label.

    :param Y: Labels (typically ground truth for train data) provided as a matrix with non-zero values for relevant labels.
    :type Y: ndarray, csr_matrix, list[list[int]]
    :return: Array with the inverse prior probabilities of labels.
    :rtype: ndarray
    """
    return 1.0 / labels_priors(Y)


def Jain_et_al_propensity(Y, A=0.55, B=1.5):
    """
    Calculate propensity as proposed in Jain et al. 2016.
    Propensity :math:`p_l` of label :math:`l` is calculated as:

    .. math::

        C = (\\log N - 1)(B + 1)^A \\,, \\
        p_l = \\frac{1}{1 + C(N_l + B)^{-A}} \\,,

    where :math:`N` is total number of data points, :math:`N_j` is total number of data points for
    and :math:`A` and :math:`B` are dataset specific parameters.

    :param Y: Labels (typically ground truth for train data) provided as a matrix with non-zero values for relevant labels.
    :type Y: ndarray, csr_matrix, list[list[int]]
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
    :return: Array with the propensity for all labels
    :rtype: ndarray
    """
    return 1.0 / Jain_et_al_inverse_propensity(Y, A, B)


def Jain_et_al_inverse_propensity(Y, A=0.55, B=1.5):
    """
    Calculate inverse propensity as proposed in Jain et al. 2016.
    Inverse propensity :math:`q_l` of label :math:`l` is calculated as:

    .. math::

        C = (\\log N - 1)(B + 1)^A \\,, \\
        q_l = 1 + C(N_l + B)^{-A} \\,,

    where :math:`N` is total number of data points, :math:`N_j` is total number of data points for
    and :math:`A` and :math:`B` are dataset specific parameters.

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
    counts = count_labels(Y)
    if isinstance(Y, (np.ndarray, csr_matrix)):
        n = Y.shape[0]
    elif isinstance(Y, list):
        n = len(Y)
    else:
        raise TypeError("Unsupported data type, should be Numpy matrix, Scipy sparse matrix or list of lists of ints")

    C = (log(n) - 1) * (B + 1) ** A
    inv_ps = 1 + C * (counts + B) ** -A
    return inv_ps


def psprecision_at_k(Y_true, Y_pred, inv_ps, k=5, normalize=True):
    """
    Calculate Propensity Scored Precision (PSP) at 1-k places.
    This measure can be also called weighted precision.
    PSP at k is defined as:

    .. math::

        psp@k = \\frac{1}{k} \\sum_{l \\in \\text{rank}_k(\\hat{\\pmb{y}})} q_l \\hat{y_l},

    where :math:`\\pmb{y} \\in {0, 1}^m` is ground truth label vector,
    :math:`\\hat{\\pmb{y}} \\in \\mathbb{R}^m` is predicted labels score vector,
    :math:`\\text{rank}_k(\\hat{\\pmb{y}})` returns the :math:`k` indices of :math:`\\hat{\\pmb{y}}`
    with the largest values, ordered in descending order,
    and :math:`\\pmb{q}` is vector of inverse propensities.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Inverse propensity (propensity scores) for each label (label weights). In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param normalize: Normalize result to [0, 1] range by dividing it by best possible value, commonly used to report results, defaults to True
    :type normalize: bool, optional
    :return: Values of PSP at 1-k places.
    :rtype: ndarray
    """
    return PSPrecisionAtK(inv_ps, k=k, normalize=normalize).calculate(Y_true, Y_pred)


def psrecall_at_k(Y_true, Y_pred, inv_ps, k=5, normalize=True, zero_division=0):
    """
    Calculate Propensity Scored Recall (PSR) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Inverse propensity (propensity scores) for each label. In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param zero_division: Value to add when there is a zero division, typically set to 0, defaults to 0
    :type zero_division: float, optional
    :param normalize: Normalize result to [0, 1] range by dividing it by best possible value, commonly used to report results, defaults to True
    :type normalize: bool, optional
    :return: Values of PSR at 1-k places.
    :rtype: ndarray
    """
    return PSRecallAtK(inv_ps, k=k, normalize=normalize, zero_division=zero_division).calculate(Y_true, Y_pred)


def psdcg_at_k(Y_true, Y_pred, inv_ps, k=5, normalize=True):
    """
    Calculate Propensity Scored Discounted Cumulative Gain (PSDCG) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Inverse propensity (propensity scores) for each label. In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param normalize: Normalize result to [0, 1] range by dividing it by best possible value, commonly used to report results, defaults to True
    :type normalize: bool, optional
    :return: Values of PSDCG at 1-k places.
    :rtype: ndarray
    """
    return PSDCGAtK(inv_ps, k=k, normalize=normalize).calculate(Y_true, Y_pred)


def psndcg_at_k(Y_true, Y_pred, inv_ps, k=5, zero_division=0, normalize=True):
    """
    Calculate Propensity Scored normalized Discounted Cumulative Gain (PSnDCG) at 1-k places.

    :param Y_true: Ground truth provided as a matrix with non-zero values for true labels or a list of lists or sets of true labels.
    :type Y_true: ndarray, csr_matrix, list[list|set[int|str]]
    :param Y_pred:
        Predicted labels provided as a matrix with scores or list of rankings as a list of labels or tuples of labels with scores (label, score).
        In the case of the matrix, the ranking will be calculated by sorting scores in descending order.
    :type Y_pred: ndarray, csr_matrix, list[list[int|str]], list[list[tuple[int|str, float]]
    :param inv_ps: Inverse propensity (propensity scores) for each label. In case of text labels needs to be a dict.
    :type inv_ps: ndarray, list, dict
    :param k: Calculate at places from 1 to k, defaults to 5
    :type k: int, optional
    :param zero_division: Value to add when there is a zero division, typically set to 0, defaults to 0
    :type zero_division: float, optional
    :param normalize: Normalize result to [0, 1] range by dividing it by best possible value, commonly used to report results, defaults to True
    :type normalize: bool, optional
    :return: Values of PSnDCG at 1-k places.
    :rtype: ndarray
    """
    return PSNDCGAtK(inv_ps, k=k, normalize=normalize, zero_division=zero_division).calculate(Y_true, Y_pred)


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
    if average == 'micro':
        return MicroF1Metric().calculate(Y_true, Y_pred)

    elif average == 'macro':
        return MacroF1Metric(zero_division=zero_division).calculate(Y_true, Y_pred)

    elif average == 'samples':
        return SamplesF1Metric(zero_division=zero_division).calculate(Y_true, Y_pred)

    else:
        raise ValueError("average should be in {'micro', 'macro', 'samples'}")


def micro_f1_measure(Y_true, Y_pred):
    return MicroF1Metric().calculate(Y_true, Y_pred)


def macro_f1_measure(Y_true, Y_pred, zero_division=0):
    return MacroF1Metric(zero_division=zero_division).calculate(Y_true, Y_pred)


def samples_f1_measure(Y_true, Y_pred, zero_division=0):
    return SamplesF1Metric(zero_division=zero_division).calculate(Y_true, Y_pred)


def macro_precision_at_k(Y_true, Y_pred, k=5, zero_division=0):
    return MacroPrecisionAtK(k=k, zero_division=zero_division).calculate(Y_true, Y_pred)


def macro_recall_at_k(Y_true, Y_pred, k=5, zero_division=0):
    return MacroRecallAtK(k=k, zero_division=zero_division).calculate(Y_true, Y_pred)


def macro_f1_measure_at_k(Y_true, Y_pred, k=5, zero_division=0):
    return MacroF1MetricAtK(k=k, zero_division=zero_division).calculate(Y_true, Y_pred)

