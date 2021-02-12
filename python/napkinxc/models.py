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

import re
from numpy import ndarray
from scipy.sparse import csr_matrix
from ._napkinxc import CPPModel, InputDataType


class Model():
    """
    Main model class that wraps CPPModel
    """

    def __init__(self, **params):
        self._model = CPPModel() # In the feature more CPP classes may be available
        self._params = {}
        self.set_params(**params)

    def fit(self, X, Y):
        """
        Fit the model to the given training data.

        :param X: Training data points as a matrix or list of lists of int or tuples of int and float (feature id, value).
        :type X: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
        :param Y: Target labels as list of ints (multi-class data) or lists or tuples of ints (multi-label data).
        :type Y: list[int], list[list|tuple[int]]
        """
        self._model.fit(X, Y, Model._check_data_type(X), Model._check_data_type(Y))

    def fit_on_file(self, path):
        """
        Fit the model to the training data in the given file in multi-label svmlight/libsvm format.

        :param path: Path to the file.
        :type path: str
        """
        self._model.fitOnFile(path)

    def load(self):
        """
        Load the model to RAM.
        """
        self._model.load()

    def unload(self):
        """
        Unload the model from RAM.
        """
        self._model.unload()

    def predict(self, X, top_k=0, threshold=0, labels_weights=None):
        """
        Predict labels for data points in X.

        :param X: Data points as a matrix or list of lists of int or tuples of int and float (feature id, value).
        :type X: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
        :param top_k: Predict top-k labels, if 0, the option is ignored, defaults to 0
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold in case of single value
            or above the specific threshold for each label in case of list or array of values,
            if 0, the option is ignored, defaults to 0
        :type threshold: float, list[float], ndarray, optional
        :param labels_weights: Predict labels according to their weights multiplied by probability
            if None, the option is ignored, defaults to None
        :type labels_weights: list[float], ndarray, optional
        :return: List of lists with predicted labels.
        :rtype: list[list[int]]
        """
        threshold = self._prepare_pred(top_k, threshold, labels_weights)
        return self._model.predict(X, Model._check_data_type(X), top_k, threshold)

    def predict_proba(self, X, top_k=0, threshold=0, labels_weights=None):
        """
        Predict labels with probability estimates for data points in X.

        :param X: Data points as a matrix or list of lists of int or tuples of int and float (feature id, value).
        :type X: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
        :param top_k: Predict top-k labels, if 0, the option is ignored, defaults to 0
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold in case of single value
            or above the specific threshold for each label in case of list or array of values,
            if 0, the option is ignored, defaults to 0
        :type threshold: float, list[float], ndarray, optional
        :param labels_weights: Predict labels according to their weights multiplied by probability
            if None, the option is ignored, defaults to None
        :type labels_weights: list[float], ndarray, optional
        :return: List of list of tuples (label id, probability) with predicted labels
        :rtype: list[list[tuple[int, float]]
        """
        threshold = self._prepare_pred(top_k, threshold, labels_weights)
        return self._model.predict_proba(X, Model._check_data_type(X), top_k, threshold)

    def predict_for_file(self, path, top_k=0, threshold=0, labels_weights=None):
        """
        Predict labels for data points in the given file in multi-label svmlight/libsvm format.

        :param path: Path to the file
        :type path: str
        :param top_k: Predict top-k labels, if 0, the option is ignored, defaults to 0
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold in case of single value
            or above the specific threshold for each label in case of list or array of values,
            if 0, the option is ignored, defaults to 0
        :type threshold: float, list[float], ndarray, optional
        :param labels_weights: Predict labels according to their weights multiplied by probability
            if None, the option is ignored, defaults to None
        :type labels_weights: list[float], ndarray, optional
        :return: List of lists with predicted labels.
        :rtype: list[list[int]]
        """
        threshold = self._prepare_pred(top_k, threshold, labels_weights)
        return self._model.predict_for_file(path, top_k, threshold)

    def predict_proba_for_file(self, path, top_k=0, threshold=0, labels_weights=None):
        """
        Predict labels with probability estimates for data points in the given file in multi-label svmlight/libsvm format.

        :param path: Path to the file.
        :type path: str
        :param top_k: Predict top-k labels, if 0, the option is ignored, defaults to 0
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold in case of single value
            or above the specific threshold for each label in case of list or array of values,
            if 0, the option is ignored, defaults to 0
        :type threshold: float, list[float], ndarray, optional
        :param labels_weights: Predict labels according to their weights multiplied by probability
            if None, the option is ignored, defaults to None
        :type labels_weights: list[float], ndarray, optional
        :return: List of list of tuples (label id, probability) with predicted labels
        :rtype: list[list[tuple[int, float]]
        """
        threshold = self._prepare_pred(top_k, threshold, labels_weights)
        return self._model.predict_proba_for_file(path, top_k, threshold)

    def ofo(self, X, Y, type='micro', a=10, b=20, epochs=1):
        """
        Perform Online F-measure Optimization procedure on the given data to find optimal thresholds.

        :param X: Data points as a matrix or list of lists of int or tuples of int and float (feature id, value).
        :type X: ndarray, csr_matrix, list[list[int]], list[list[tuple[int, float]]
        :param Y: Target labels as list of ints (multi-class data) or lists or tuples of ints (multi-label data).
        :type Y: list[int], list[list|tuple[int]]
        :param type: Type of OFO procedure {``'micro'``, ``'macro'``}, default to ``'micro'``
        :type type: str
        :param a: Parameter of OFO procedure, defaults to 10
        :type a: int
        :param b: Parameter of OFO procedure, defaults to 20
        :type b: int
        :param epochs: Number of OFO epochs, defaults to 1
        :type epochs: int, optional
        :return: Single threshold in case of ``type='micro'`` and list of thresholds in case of ``type='macro'``
        :rtype: float, list[float]
        """
        self.set_params(ofo_type=type, ofo_a=a, ofo_b=b, epochs=epochs)
        thr = self._model.ofo(X, Y, Model._check_data_type(X), Model._check_data_type(Y))
        if type == 'micro':
            thr = thr[0]
        return thr

    def get_params(self, deep=False): # deep argument for Scikit-learn compatibility
        """
        Get parameters of this model.

        :param deep: Ignored, added for Scikit-learn compatibility, defaults to False
        :return: Mapping of string to any
        :rtype: dict
        """
        return self._params

    def set_params(self, **params):
        """
        Set parameters for this model.

        :param: \*\*params: Parameter names with their new values.
        :return: self
        :rtype: Model
        """
        if 'model' in self._params and 'model' in params:
            params.pop('model')  # do not allow for changing model param

        if 'verbose' in params and params['verbose'] == True:
            params['verbose'] = 2  # override verbose

        self._params.update(params)
        params_list = []
        for k, v in params.items():
            if v is None:
                continue

            arg_k = ("--" if len(k) > 1 else "-") + Model._to_camelcase(k)
            arg_v = v
            if type(arg_v) == bool:
                arg_v = int(arg_v)
            arg_v = str(arg_v)

            params_list.extend([arg_k, arg_v])
        self._model.set_args(params_list)

        return self

    @staticmethod
    def _get_init_params(locals):
        kwargs = locals['kwargs']
        for k in ['self', '__class__', 'kwargs']:
            locals.pop(k)
        locals.update(kwargs)
        return locals

    @staticmethod
    def _to_camelcase(string):
        return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), string)

    @staticmethod
    def _check_data_type(data):
        if isinstance(data, list):
            return InputDataType.list
        elif isinstance(data, ndarray):
            return InputDataType.ndarray
        elif isinstance(data, csr_matrix):
            return InputDataType.csr_matrix
        else:
            return -1

    def _prepare_pred(self, top_k, threshold, labels_weights):
        if top_k == 0 and threshold == 0:
            print("Warning: both top_k and threshold arguments set to 0, this will predict all labels")

        if not isinstance(top_k, int):
            raise TypeError("Unsupported top_k type, should be int")

        if isinstance(threshold, (list, ndarray)):
            self._model.set_thresholds(threshold)
            threshold = 0
        elif not isinstance(threshold, (float, int)):
            raise TypeError("Unsupported threshold type, should be float, or list of floats, or Numpy vector (1d array)")

        if isinstance(labels_weights, (list, ndarray)):
            self._model.set_labels_weights(labels_weights)
        elif labels_weights is not None:
            raise TypeError("Unsupported labels_weights type, should be list of floats, or Numpy vector (1d array)")

        return threshold


class PLT(Model):
    """
    Probabilistic Labels Trees (PLTs) (multi-label) classifier with linear node estimators, using CPP core.
    """

    def __init__(self,
                 output,

                 # Tree params
                 tree_type='hierarchicalKmeans',
                 arity=2,
                 max_leaves=100,
                 kmeans_eps=0.0001,
                 kmeans_balanced=True,
                 #tree_structure=None, #TODO

                 # Features params
                 hash=None,
                 features_threshold=0,
                 norm=True,
                 bias=1.0,

                 # Base (node) classifiers params
                 optimizer='liblinear',
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
                 liblinear_solver=None,
                 liblinear_max_iter=100,
                 eta=1.0,
                 epochs=1,
                 adagrad_eps=0.001,

                 # Other
                 ensemble=1,
                 seed=None,
                 threads=0,
                 verbose=0,
                 **kwargs):
        """
        Construct a Probabilistic Labels Trees model.

        :param output: Directory where the model will be stored
        :type output: str
        :param tree_type: Tree type to construct {``'hierarchicalKmeans'``, ``'balancedRandom'``, ``'completeKaryRandom'``, ``'huffman'``}, defaults to ``'hierarchicalKmeans'``
        :type tree_type: str, optional
        :param arity: Arity of tree nodes, k for k-means clustering used in hierarchical k-means tree building procedure, defaults to 2
        :type arity: int, optional
        :param max_leaves: Maximum degree of pre-leaf nodes, defaults to 100
        :type max_leaves: int, optional
        :param kmeans_eps: Tolerance of termination criterion of the k-means clustering used in hierarchical k-means tree building procedure, defaults to 0.0001
        :type kmeans_eps: float, optional
        :param kmeans_balanced: Use balanced k-means clustering, defaults to True
        :type kmeans_balanced: bool, optional
        :param hash: Hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: Prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: Unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: Value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param optimizer: Optimizer used for training node classifiers {``'liblinear'``, ``'sgd'``, ``'adagrad'``}, defaults to ``'liblinear'``
        :type optimizer: str, optional
        :param loss: Loss optimized while training node classifiers {``'log'`` (alias ``'logistic'``), ``'l2'`` (alias ``'squaredHinge'``)}, defaults to ``'log'``
        :type loss: str, optional
        :param weights_threshold: Threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, makes effect only if ``optimizer='liblinear'``, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, makes effect only if ``optimizer='liblinear'``, defaults to 0.1
        :type liblinear_eps: float, optional
        :param liblinear_solver: Override LIBLINEAR solver set by loss parameter (default for ``loss='log'``: ``'L2R_LR_DUAL'``, for ``loss='l2'``: ``'L2R_L2LOSS_SVC_DUAL'``), makes effect only if ``optimizer='liblinear'``.
            Available solvers:

            - ``'L2R_LR_DUAL'``
            - ``'L2R_LR'``
            - ``'L1R_LR'``
            - ``'L2R_L2LOSS_SVC_DUAL'``
            - ``'L2R_L2LOSS_SVC'``
            - ``'L2R_L1LOSS_SVC_DUAL'``
            - ``'L1R_L2LOSS_SVC'``

            ``L2R_LR_DUAL`` and ``L2R_L2LOSS_SVC_DUAL`` usually work the best in XC setting, defaults to None
        :type liblinear_solver: str, optional
        :param liblinear_max_iter: Limits number of iteration by LIBLINEAR, makes effect only if ``optimizer='liblinear'``, defaults to 100
        :type liblinear_max_iter: int, optional
        :param eta: Step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: Number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: Defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param ensemble: Number of trees in the ensemble, defaults to 1
        :type ensemble: int, optional
        :param seed: Seed, If None use current system time, defaults to None
        :type seed: int, optional
        :param threads: Number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param verbose: If True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "plt"})
        super(PLT, self).__init__(**all_params)


class HSM(Model):
    """
    Hierarchical Softmax (multi-class) classifier with linear node estimators, using CPP core.
    """

    def __init__(self,
                 output,

                 # Tree params
                 tree_type='hierarchicalKmeans',
                 arity=2,
                 max_leaves=100,
                 kmeans_eps=0.0001,
                 kmeans_balanced=True,
                 #tree_structure=None, #TODO

                 # Features params
                 hash=None,
                 features_threshold=0,
                 norm=True,
                 bias=1.0,
                 pick_one_label_weighting=False,

                 # Base (node) classifiers params
                 optimizer='liblinear',
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
                 liblinear_solver=None,
                 liblinear_max_iter=100,
                 eta=1.0,
                 epochs=1,
                 adagrad_eps=0.001,

                 # Other
                 ensemble=1,
                 seed=None,
                 threads=0,
                 verbose=0,
                 **kwargs):
        """
        Construct a Hierarchical Softmax model.

        :param output: Directory where the model will be stored
        :type output: str
        :param tree_type: Tree type to construct {``'hierarchicalKmeans'``, ``'balancedRandom'``, ``'completeKaryRandom'``, ``'huffman'``}, defaults to ``'hierarchicalKmeans'``
        :type tree_type: str, optional
        :param arity: Arity of tree nodes, k for k-means clustering used in hierarchical k-means tree building procedure, defaults to 2
        :type arity: int, optional
        :param max_leaves: Maximum degree of pre-leaf nodes, defaults to 100
        :type max_leaves: int, optional
        :param kmeans_eps: Tolerance of termination criterion of the k-means clustering used in hierarchical k-means tree building procedure, defaults to 0.0001
        :type kmeans_eps: float, optional
        :param kmeans_balanced: Use balanced k-means clustering, defaults to True
        :type kmeans_balanced: bool, optional
        :param hash: Hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: Prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: Unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: Value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param optimizer: Optimizer used for training node classifiers {``'liblinear'``, ``'sgd'``, ``'adagrad'``}, defaults to ``'liblinear'``
        :type optimizer: str, optional
        :param loss: Loss optimized while training node classifiers {``'log'`` (alias ``'logistic'``), ``'l2'`` (alias ``'squaredHinge'``)}, defaults to ``'log'``
        :type loss: str, optional
        :param weights_threshold: Threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, makes effect only if ``optimizer='liblinear'``, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, makes effect only if ``optimizer='liblinear'``, defaults to 0.1
        :type liblinear_eps: float, optional
        :param liblinear_solver: Override LIBLINEAR solver set by loss parameter (default for ``loss='log'``: ``'L2R_LR_DUAL'``, for ``loss='l2'``: ``'L2R_L2LOSS_SVC_DUAL'``), makes effect only if ``optimizer='liblinear'``.
            Available solvers:

            - ``'L2R_LR_DUAL'``
            - ``'L2R_LR'``
            - ``'L1R_LR'``
            - ``'L2R_L2LOSS_SVC_DUAL'``
            - ``'L2R_L2LOSS_SVC'``
            - ``'L2R_L1LOSS_SVC_DUAL'``
            - ``'L1R_L2LOSS_SVC'``

            ``L2R_LR_DUAL`` and ``L2R_L2LOSS_SVC_DUAL`` usually work the best in XC setting, defaults to None
        :type liblinear_solver: str, optional
        :param liblinear_max_iter: Limits number of iteration by LIBLINEAR, makes effect only if ``optimizer='liblinear'``, defaults to 100
        :type liblinear_max_iter: int, optional
        :param eta: Step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: Number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: Defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param ensemble: Number of trees in the ensemble, defaults to 1
        :type ensemble: int, optional
        :param seed: Seed, If None use current system time, defaults to None
        :type seed: int, optional
        :param threads: Number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param verbose: If True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "hsm"})
        super(HSM, self).__init__(**all_params)


class BR(Model):
    """
    Binary Relevance (multi-label) classifier with linear estimators, using CPP core
    """

    def __init__(self,
                 output,

                 # Features params
                 hash=None,
                 features_threshold=0,
                 norm=True,
                 bias=1.0,

                 # Base classifiers params
                 optimizer='liblinear',
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
                 liblinear_solver=None,
                 liblinear_max_iter=100,
                 eta=1.0,
                 epochs=1,
                 adagrad_eps=0.001,

                 # Other
                 threads=0,
                 mem_limit=0,
                 verbose=0,
                 **kwargs):
        """
        Construct a Binary Relevance model.

        :param output: Directory where the model will be stored
        :type output: str
        :param hash: Hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: Prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: Unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: Value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param optimizer: Optimizer used for training node classifiers {``'liblinear'``, ``'sgd'``, ``'adagrad'``}, defaults to ``'liblinear'``
        :type optimizer: str, optional
        :param loss: Loss optimized while training node classifiers {``'log'`` (alias ``'logistic'``), ``'l2'`` (alias ``'squaredHinge'``)}, defaults to ``'log'``
        :type loss: str, optional
        :param weights_threshold: Threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, makes effect only if ``optimizer='liblinear'``, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, makes effect only if ``optimizer='liblinear'``, defaults to 0.1
        :type liblinear_eps: float, optional
        :param liblinear_solver: Override LIBLINEAR solver set by loss parameter (default for ``loss='log'``: ``'L2R_LR_DUAL'``, for ``loss='l2'``: ``'L2R_L2LOSS_SVC_DUAL'``), makes effect only if ``optimizer='liblinear'``.
            Available solvers:

            - ``'L2R_LR_DUAL'``
            - ``'L2R_LR'``
            - ``'L1R_LR'``
            - ``'L2R_L2LOSS_SVC_DUAL'``
            - ``'L2R_L2LOSS_SVC'``
            - ``'L2R_L1LOSS_SVC_DUAL'``
            - ``'L1R_L2LOSS_SVC'``

            ``L2R_LR_DUAL`` and ``L2R_L2LOSS_SVC_DUAL`` usually work the best in XC setting, defaults to None
        :type liblinear_solver: str, optional
        :param liblinear_max_iter: Limits number of iteration by LIBLINEAR, makes effect only if ``optimizer='liblinear'``, defaults to 100
        :type liblinear_max_iter: int, optional
        :param eta: Step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: Number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: Defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param threads: Number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param mem_limit: Maximum amount of memory (in G) available for training, if 0 use amount of available memory, defaults to 0
        :type mem_limit: float
        :param verbose: If True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "br"})
        super(BR, self).__init__(**all_params)


class OVR(Model):
    """
    One Versus Rest (multi-class) classifier with linear estimators, using CPP core.
    """

    def __init__(self,
                 output,

                 # Features params
                 hash=None,
                 features_threshold=0,
                 norm=True,
                 bias=1.0,
                 pick_one_label_weighting=False,

                 # Base classifiers params
                 optimizer='liblinear',
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
                 liblinear_solver=None,
                 liblinear_max_iter=100,
                 eta=1.0,
                 epochs=1,
                 adagrad_eps=0.001,

                 # Other
                 threads=0,
                 mem_limit=0,
                 verbose=0,
                 **kwargs):
        """
        Construct a Multi-class One Versus Rest model.

        :param output: Directory where the model will be stored
        :type output: str
        :param hash: Hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: Prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: Unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: Value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param pick_one_label_weighting: Allows to use multi-label data by transforming it into multi-class, defaults to False
        :type pick_one_label_weighting: bool, optional
        :param optimizer: Optimizer used for training node classifiers {``'liblinear'``, ``'sgd'``, ``'adagrad'``}, defaults to ``'liblinear'``
        :type optimizer: str, optional
        :param loss: Loss optimized while training node classifiers {``'log'`` (alias ``'logistic'``), ``'l2'`` (alias ``'squaredHinge'``)}, defaults to ``'log'``
        :type loss: str, optional
        :param weights_threshold: Threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, makes effect only if ``optimizer='liblinear'``, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, makes effect only if ``optimizer='liblinear'``, defaults to 0.1
        :type liblinear_eps: float, optional
        :param liblinear_solver: Override LIBLINEAR solver set by loss parameter (default for ``loss='log'``: ``'L2R_LR_DUAL'``, for ``loss='l2'``: ``'L2R_L2LOSS_SVC_DUAL'``), makes effect only if ``optimizer='liblinear'``.
            Available solvers:

            - ``'L2R_LR_DUAL'``
            - ``'L2R_LR'``
            - ``'L1R_LR'``
            - ``'L2R_L2LOSS_SVC_DUAL'``
            - ``'L2R_L2LOSS_SVC'``
            - ``'L2R_L1LOSS_SVC_DUAL'``
            - ``'L1R_L2LOSS_SVC'``

            ``L2R_LR_DUAL`` and ``L2R_L2LOSS_SVC_DUAL`` usually work the best in XC setting, defaults to None
        :type liblinear_solver: str, optional
        :param liblinear_max_iter: Limits number of iteration by LIBLINEAR, makes effect only if ``optimizer='liblinear'``, defaults to 100
        :type liblinear_max_iter: int, optional
        :param eta: Step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: Number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: Defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param threads: Number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param mem_limit: Maximum amount of memory (in G) available for training, if 0 use amount of available memory, defaults to 0
        :type mem_limit: float
        :param verbose: If True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "ovr"})
        super(OVR, self).__init__(**all_params)
