from _napkinxc import CPPModel


def _get_init_params(locals):
    kwargs = locals['kwargs']
    for k in ['self', '__class__', 'kwargs']:
        locals.pop(k)
    locals.update(kwargs)
    return locals


class Model():
    """
    Main model class that wraps CPPModel
    """

    def __init__(self, **params):
        self._model = CPPModel() # In the feature more CPP classes may be available
        self._params = params
        self.set_params(**self._params)

    def fit(self, X, y):
        self._model.fit(X, y, 1, 1)

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

        return self._model.predict(X, 1)

    def predict_for_file(self, path):
        return self._model.predictForFile(path)

    def get_params(self, deep=True): # deep argument for sklearn compatibility
        return self._params

    def set_params(self, **params):
        self._params.update(params)
        params_list = []
        for k,v in params.items():
            params_list.extend(["--" if len(str(k)) > 1 else "-" + str(k), str(v)])
        self._model.set_args(params_list)


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
                 hash=0,
                 features_threshold=0,
                 norm=True,
                 bias=1.0,

                 # Node classifiers params
                 optimizer="liblinear",
                 loss='log',
                 weights_threshold=0.1,
                 liblinear_eps=0.1,
                 eta=1.0,
                 epochs=1,
                 adagrad_eps=0.01,

                 # Other
                 ensemble=1,
                 seed=0,
                 threads=0,
                 **kwargs):
        """
        Initialize Probabilistic Labels Trees
        :param tree_type:
        :param arity:
        :param max_leaves:
        :param kmeans_eps:
        :param kmeans_balanced:
        :param tree_structure:
        :param hash: hash features to a space of given size, 0 to disable hashing (default = 0)
        :param features_threshold: prune features belowe given threshold (default = 0)
        :param norm:
        :param bias:
        :param optimizer:
        :param loss:
        :param weights_threshold:
        :param liblinear_eps:
        :param eta:
        :param epochs:
        :param adagrad_eps:
        :param ensemble:
        :param seed:
        :param threads:
        :return: None
        """

        all_params = _get_init_params(locals())
        all_params.update({"model": "plt"})
        super(PLT, self).__init__(**all_params)
