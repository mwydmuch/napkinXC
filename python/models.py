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
        self.set_params(**self._params)

    def fit(self, X, y):
        self._model.fit(X, y, Model._get_data_type(X), Model._get_data_type(y))

    def fit_on_file(self, path):
        self._model.fitOnFile(path)

    def predict(self, X, top_k=5, threshold=0, thresholds=None):
        """
        Predict labels with probability estimates
        :param X:
        :param top_k: Predict top k elements (default = 5)
        :param threshold:
        :param thresholds:
        :return:
        """

        return self._model.predict(Model._get_data_type(X))

    def predict_for_file(self, path):
        return self._model.predictForFile(path)

    def get_params(self, deep=True): # deep argument for sklearn compatibility
        return self._params

    def set_params(self, **params):
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
    def _get_data_type(data):
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
    Probabilistic Labels Trees model with linear node estimators, using CPP core
    """

    def __init__(self,
                 # Tree params
                 tree_type='k-means',
                 arity=2,
                 max_leaves=100,
                 kmeans_eps=0.001,
                 kmeans_balanced=True,
                 tree_structure=None,

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
                 adagrad_eps=0.01,

                 # Other
                 ensemble=1,
                 seed=None,
                 threads=0,
                 **kwargs):
        """
        Initialize Probabilistic Labels Trees
        :param tree_type:
        :param arity: arity of tree nodes, k for k-means clustering used in hierarchical k-means tree building procedure (default = 2)
        :param max_leaves: maximum degree of pre-leaf nodes
        :param kmeans_eps: tolerance of termination criterion of the k-means clustering used in hierarchical k-means tree building procedure (default = 0.0001)
        :param kmeans_balanced: perform balanced k-means clustering (default = True)
        :param tree_structure:
        :param hash: hash features to a space of given size, None or 0 disables hashing (default = None)
        :param features_threshold: prune features belowe given threshold (default = 0)
        :param norm:
        :param bias:
        :param optimizer: optimizer used for training node classifiers {'liblinear', 'sgd', 'adagrad'}, (default = 'libliner')
        :param loss: loss optimized while training node classifiers {'logistic', 'l2' (squared hinge)}, (default = 'logistic')
        :param weights_threshold: threshold value for pruning weights (default = 0.1)
        :param liblinear_c: LIBLINEAR cost co-efficient, inverse regularization strength, smaller values specify stronger regularization (default = 10)
        :param liblinear_eps: LIBLINEAR tolerance of termination criterion (default = 0.1)
        :param eta:
        :param epochs: number of epochs taken for the solvers to converge
        :param adagrad_eps:
        :param ensemble: number of trees in ensemble
        :param seed:
        :param threads:
        :return: None
        """

        all_params = Model._get_init_params(locals())
        all_params.update({"model": "plt"})
        super(PLT, self).__init__(**all_params)
