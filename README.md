# napkinXC 
[![C++ build](https://github.com/mwydmuch/napkinXC/workflows/C++%20build/badge.svg)](https://github.com/mwydmuch/napkinXC/actions/workflows/cpp-build.yml)
[![Python build](https://github.com/mwydmuch/napkinXC/workflows/Python%20build/badge.svg)](https://github.com/mwydmuch/napkinXC/actions/workflows/python-build.yml)
[![Documentation Status](https://readthedocs.org/projects/napkinxc/badge/?version=latest)](https://napkinxc.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/napkinxc.svg)](https://badge.fury.io/py/napkinxc) 

napkinXC is an extremely simple and fast library for extreme multi-class and multi-label classification, 
that focus on implementing various methods for Probabilistic Label Trees.
It allows training a classifier for very large datasets in just a few lines of code with minimal resources.

Right now, napkinXC implements the following features both in Python and C++:
- Probabilistic Label Trees (PLTs) and Hierarchical softmax (HSM),
- different types of inference methods (top-k, above a given threshold, etc.),
- fast prediction with labels weight, e.g., propensity scores,
- efficient online F-measure optimization (OFO) procedure,
- different tree building methods, including hierarchical k-means clustering method,
- training of tree node
- support for custom tree structures, and node weights, 
- helpers to download and load data from [XML Repository](http://manikvarma.org/downloads/XC/XMLRepository.html),
- helpers to measure performance (precision@k, recall@k, nDCG@k, propensity-scored precision@k, and more).

Please note that this library is still under development and also serves as a base for experiments.
API may not be compatible between releases and some of the experimental features may not be documented.
Do not hesitate to open an issue in case of a question or problem!

The napkinXC is distributed under the MIT license. 
All contributions to the project are welcome!


## Python Quick Start and Documentation

Install via pip:
```
pip install napkinxc
```
We provide precompiled wheels for many Linux distros, macOS, and Windows for Python 3.7+.
In case there is no wheel for your os, it will be quickly compiled from the source.
Compilation from source requires modern C++17 compiler, CMake, Git, and Python 3.7+ installed.


The latest (master) version can be installed directly from the GitHub repository (not recommended):
```
pip install git+https://github.com/mwydmuch/napkinXC.git
```


A minimal example of usage:
```
from napkinxc.datasets import load_dataset
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k

X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")
plt = PLT("eurlex-model")
plt.fit(X_train, Y_train)
Y_pred = plt.predict(X_test, top_k=1)
print(precision_at_k(Y_test, Y_pred, k=1)) 
```

More examples can be found under [`python/examples directory`](https://github.com/mwydmuch/napkinXC/tree/master/python/examples),
and napkinXC's documentation is available at [https://napkinxc.readthedocs.io](https://napkinxc.readthedocs.io).


## Executable

napkinXC can also be used as executable to train and evaluate models using data in LIBSVM format.
See [documentation](https://napkinxc.readthedocs.io/en/latest/exe_usage.html) for more details.


## References and acknowledgments

This library implements methods from the following papers (see `experiments` directory for scripts to replicate the results):

- [Probabilistic Label Trees for Extreme Multi-label Classification](https://arxiv.org/abs/2009.11218)
- [Online probabilistic label trees](http://proceedings.mlr.press/v130/jasinska-kobus21a.html)
- [Propensity-scored Probabilistic Label Trees](https://dl.acm.org/doi/10.1145/3404835.3463084)
- [Efficient Algorithms for Set-Valued Prediction in Multi-Class Classification](https://link.springer.com/article/10.1007/s10618-021-00751-x)

Another implementation of PLT model is available in [extremeText](https://github.com/mwydmuch/extremeText) library, 
that implements approach described in this [NeurIPS paper](http://papers.nips.cc/paper/7872-a-no-regret-generalization-of-hierarchical-softmax-to-extreme-multi-label-classification).
