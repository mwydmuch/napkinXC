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
        Fit the model to the given training data

        :param X: training data points
        :type X: array-like, sparse matrix, list of lists of tuples (idx, value)
        :param Y: target labels
        :type Y: list of lists or tuples
        :return: None
        """
        self._model.fit(X, Y, Model._check_data_type(X), Model._check_data_type(Y))

    def fit_on_file(self, path):
        """
        Fit the model to the training data in the given file

        :param path: path to the file
        :type path: str
        :return: None
        """
        self._model.fitOnFile(path)

    def predict(self, X, top_k=5, threshold=0):
        """
        Predict labels for data points in X

        :param X: data points
        :type X: array-like, sparse matrix, list of lists of tuples (idx, value)
        :param top_k: Predict top-k labels, defaults to 5
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold, defaults to 0
        :type threshold: float
        :return: list of list
        """
        return self._model.predict(X, Model._check_data_type(X), top_k, threshold)

    def predict_proba(self, X, top_k=5, threshold=0):
        """
        Predict labels with probability estimates for data points in X

        :param X: data points
        :type X: array-like, sparse matrix, list of lists of tuples (idx, value)
        :param top_k: Predict top-k labels, defaults to 5
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold, defaults to 0
        :type threshold: float
        :return: list of list of tuples
        """
        return self._model.predict(X, Model._check_data_type(X), top_k, threshold)

    def predict_for_file(self, path, top_k=5, threshold=0):
        """
        Predict labels for data points in the given file

        :param path: path to the file
        :type path: str
        :param top_k: Predict top-k labels, defaults to 5
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold, defaults to 0
        :type threshold: float
        :return: list of list
        """
        return self._model.predict_for_file(path, top_k, threshold)

    def predict_proba_for_file(self, path, top_k=5, threshold=0):
        """
        Predict labels with probability estimates for data points in the given file

        :param path: path to the file
        :type path: str
        :param top_k: Predict top-k labels, defaults to 5
        :type top_k: int
        :param threshold: Predict labels with probability above the threshold, defaults to 0
        :type threshold: float
        :return: list of list of tuples
        """
        return self._model.predict_proba_for_file(path, top_k, threshold)

    def get_params(self, deep=False): # deep argument for sklearn compatibility
        """
        Get parameters of this model/

        :param deep: Ignored, added for sklearn compatibility, defaults to False.
        :return: mapping of string to any
        """
        return self._params

    def set_params(self, **params):
        """
        Set parameters for this model.

        :param: \*\*params: Parameter names with their new values.
        :return: self
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
        data_type = type(data)
        if data_type == list:
            return InputDataType.list
        elif data_type == ndarray:
            return InputDataType.ndarray
        elif data_type == csr_matrix:
            return InputDataType.csr_matrix
        else:
            return -1


class PLT(Model):
    """
    Probabilistic Labels Trees (PLTs) model with linear node estimators, using CPP core.
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

                 # Node classifiers params
                 optimizer="liblinear",
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
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

        :param output: directory where the model will be stored
        :type output: str
        :param tree_type: tree type to construct {'hierarchicalKmeans', 'balancedRandom', 'completeKaryRandom', 'huffman'}, defaults to 'hierarchicalKmeans'
        :type tree_type: str, optional
        :param arity: arity of tree nodes, k for k-means clustering used in hierarchical k-means tree building procedure, defaults to 2
        :type arity: int, optional
        :param max_leaves: maximum degree of pre-leaf nodes, defaults to 100
        :type max_leaves: int, optional
        :param kmeans_eps: tolerance of termination criterion of the k-means clustering used in hierarchical k-means tree building procedure, defaults to 0.0001
        :type kmeans_eps: float, optional
        :param kmeans_balanced: use balanced k-means clustering, defaults to True
        :type kmeans_balanced: bool, optional
        :param hash: hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param optimizer: optimizer used for training node classifiers {'liblinear', 'sgd', 'adagrad'}, defaults to 'libliner'
        :type optimizer: str, optional
        :param loss: loss optimized while training node classifiers {'logistic', 'l2' (squared hinge)}, defaults to 'logistic'
        :type loss: str, optional
        :param weights_threshold: threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, defaults to 0.1
        :type liblinear_eps: float, optional
        :param eta: step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param ensemble: number of trees in the ensemble, defaults to 1
        :type ensemble: int, optional
        :param seed: seed, if None use current system time, defaults to None
        :type seed: int, optional
        :param threads: number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param verbose: if True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "plt"})
        super(PLT, self).__init__(**all_params)


class HSM(Model):
    """
    Hierarchical Softmax model with linear node estimators, using CPP core.
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

                 # Node classifiers params
                 optimizer="liblinear",
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
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

        :param output: directory where the model will be stored
        :type output: str
        :param tree_type: tree type to construct {'hierarchicalKmeans', 'balancedRandom', 'completeKaryRandom', 'huffman'}, defaults to 'hierarchicalKmeans'
        :type tree_type: str, optional
        :param arity: arity of tree nodes, k for k-means clustering used in hierarchical k-means tree building procedure, defaults to 2
        :type arity: int, optional
        :param max_leaves: maximum degree of pre-leaf nodes, defaults to 100
        :type max_leaves: int, optional
        :param kmeans_eps: tolerance of termination criterion of the k-means clustering used in hierarchical k-means tree building procedure, defaults to 0.0001
        :type kmeans_eps: float, optional
        :param kmeans_balanced: use balanced k-means clustering, defaults to True
        :type kmeans_balanced: bool, optional
        :param hash: hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param optimizer: optimizer used for training binary classifiers {'liblinear', 'sgd', 'adagrad'}, defaults to 'libliner'
        :type optimizer: str, optional
        :param loss: loss optimized while training binary classifiers {'logistic', 'l2' (squared hinge)}, defaults to 'logistic'
        :type loss: str, optional
        :param weights_threshold: threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, defaults to 0.1
        :type liblinear_eps: float, optional
        :param eta: step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param ensemble: number of trees in the ensemble, defaults to 1
        :type ensemble: int, optional
        :param seed: seed, if None use current system time, defaults to None
        :type seed: int, optional
        :param threads: number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param verbose: if True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "hsm"})
        super(HSM, self).__init__(**all_params)


class BR(Model):
    """
    Binary Relevance model with linear node estimators, using CPP core
    """

    def __init__(self,
                 output,
                 # Features params
                 hash=None,
                 features_threshold=0,
                 norm=True,
                 bias=1.0,

                 # Node classifiers params
                 optimizer="liblinear",
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
                 eta=1.0,
                 epochs=1,
                 adagrad_eps=0.001,

                 # Other
                 threads=0,
                 verbose=0,
                 **kwargs):
        """
        Construct a Binary Relevance model.

        :param output: directory where the model will be stored
        :type output: str, optional
        :param hash: hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param optimizer: optimizer used for training binary classifiers {'liblinear', 'sgd', 'adagrad'}, defaults to 'libliner'
        :type optimizer: str, optional
        :param loss: loss optimized while training binary classifiers {'logistic', 'l2' (squared hinge)}, defaults to 'logistic'
        :type loss: str, optional
        :param weights_threshold: threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, defaults to 0.1
        :type liblinear_eps: float, optional
        :param eta: step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param threads: number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param verbose: if True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "br"})
        super(BR, self).__init__(**all_params)


class OVR(Model):
    """
    One Versus Rest model with linear node estimators, using CPP core.
    """

    def __init__(self,
                 output,
                 # Features params
                 hash=None,
                 features_threshold=0,
                 norm=True,
                 bias=1.0,

                 # Node classifiers params
                 optimizer="liblinear",
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_c=10,
                 liblinear_eps=0.1,
                 eta=1.0,
                 epochs=1,
                 adagrad_eps=0.001,

                 # Other
                 threads=0,
                 verbose=0,
                 **kwargs):
        """
        Construct a One Versus Rest model.

        :param output: directory where the model will be stored
        :type output: str
        :param hash: hash features to a space of given size, if None or 0 disable hashing, defaults to None
        :type hash: int, optional
        :param features_threshold: prune features below given threshold, defaults to 0
        :type features_threshold: float, optional
        :param norm: unit norm feature vector, defaults to True
        :type norm: bool, optional
        :param bias: value of the bias features, defaults to 1.0
        :type bias: float, optional
        :param optimizer: optimizer used for training node classifiers {'liblinear', 'sgd', 'adagrad'}, defaults to 'libliner'
        :type optimizer: str, optional
        :param loss: loss optimized while training node classifiers {'logistic', 'l2' (squared hinge)}, defaults to 'logistic'
        :type loss: str, optional
        :param weights_threshold: threshold value for pruning weights, defaults to 0.1
        :type weights_threshold: float, optional
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization, defaults to 10.0
        :type liblinear_c: float, optional
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion, defaults to 0.1
        :type liblinear_eps: float, optional
        :param eta: step size (learning rate) for online optimizers, defaults to 1.0
        :type eta: float, optional
        :param epochs: number of training epochs for online optimizers, defaults to 1
        :type epochs: int, optional
        :param adagrad_eps: defines starting step size for AdaGrad, defaults to 0.001
        :type adagrad_eps: float, optional
        :param threads: number of threads used for training and prediction, if 0 use number of available CPUs, if -1 use number of available CPUs - 1, defaults to 0
        :type threads: int, optional
        :param verbose: if True print progress, defaults to False
        :type verbose: bool, optional
        """
        all_params = Model._get_init_params(locals())
        all_params.update({"model": "ovr"})
        super(OVR, self).__init__(**all_params)
