# napkinXC 
![C++ build](https://github.com/mwydmuch/napkinXC/workflows/C++%20build/badge.svg)
![Python build](https://github.com/mwydmuch/napkinXC/workflows/Python%20build/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/napkinxc/badge/?version=latest)](https://napkinxc.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/napkinxc.svg)](https://badge.fury.io/py/napkinxc) 

napkinXC is an extremely simple and fast library for extreme multi-class and multi-label classification.
It allows training a classifier for very large datasets in just a few lines of code with minimal resources.

Right now, napkinXC implements the following features both in Python and C++:
- Probabilistic Label Trees (PLTs) and Online Probabilistic Label Trees (OPLTs),
- Hierarchical softmax (HSM),
- Binary Relevance (BR),
- One Versus Rest (OVR),
- fast online prediction of top-k labels or labels above one given threshold or separate threshold for each label,
- fast online prediction with labels weight, e.g., propensity scores,
- efficient online F-measure optimization (OFO) procedure,
- hierarchical k-means clustering for tree building and other tree-building methods,
- support for predefined hierarchies,
- LIBLINEAR, AdaGrad, SGD optimizers for base classifiers,
- ensembles of PLTs and HSMs,
- helpers to download and load data from [XML Repository](http://manikvarma.org/downloads/XC/XMLRepository.html),
- helpers to measure performance (precision@k, recall@k, nDCG@k, propensity-scored precision@k, and more).

Please note that this library is still under development and also serves as a base for experiments.
API may not be compatible between releases and some of the experimental features may not be documented.
Do not hesitate to open an issue in case of a question or problem!

The napkinXC is distributed under MIT license. 
All contributions to the project are welcome!


## Roadmap

Coming soon:

- Possibility to use any binary classifier from Python.
- Raw versions of datasets from XML Repository.


## Python Quick Start and Documentation

napkinXC's documentation is available at [https://napkinxc.readthedocs.io](https://napkinxc.readthedocs.io) 
and is generated from this repository. 

Python (3.5+) version of napkinXC can be easily installed from PyPy repository on Linux and MacOS, 
it requires modern C++17 compiler, CMake and Git installed:
```
pip install napkinxc
```

or the latest master version directly from the GitHub repository:
```
pip install git+https://github.com/mwydmuch/napkinXC.git
```

Minimal example of usage:
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

More examples can be found under [`python/examples`](https://github.com/mwydmuch/napkinXC/tree/master/python/examples) directory.


## Executable

napkinXC can also be used as executable to train and evaluate model and make a predict using a data in libsvm format

To build executable use:
```
cmake .
make
```

Command line options:

```
Usage: nxc [command] [arg...]

Commands:
    train                   Train model on given input data
    test                    Test model on given input data
    predict                 Predict for given data
    ofo                     Use online f-measure optimization
    version                 Print napkinXC version
    help                    Print help

Args:
    General:
    -i, --input             Input dataset, required
    -o, --output            Output (model) dir, required
    -m, --model             Model type (default = plt)
                            Models: ovr, br, hsm, plt, oplt, svbopFull, svbopHf, brMips, svbopMips
    --ensemble              Number of models in ensemble (default = 1)
    -t, --threads           Number of threads to use (default = 0)
                            Note: set to -1 to use a number of available CPUs - 1, 0 to use a number of available CPUs
    --memLimit              Maximum amount of memory (in G) available for training (defualt = 0)
                            Note: set to 0 to set limit to amount of available memory
    --hash                  Size of features space (default = 0)
                            Note: set to 0 to disable hashing
    --featuresThreshold     Prune features below given threshold (default = 0.0)
    --seed                  Seed (default = system time)
    --verbose               Verbose level (default = 2)

    For OVR and HSM:
    --pickOneLabelWeighting Allows to use multi-label data by transforming it into multi-class (default = 0)

    Base classifiers:
    --optim, --optimizer    Optimizer used for training binary classifiers (default = liblinear)
                            Optimizers: liblinear, sgd, adagrad
    --bias                  Value of the bias features (default = 1)
    --inbalanceLabelsWeighting      Increase the weight of minority labels in base classifiers (default = 0)
    --weightsThreshold      Threshold value for pruning models weights (default = 0.1)
    --loss                  Loss function to optimize in base classifier (default = log)
                            Losses: log (alias logistic), l2 (alias squaredHinge)

    LIBLINEAR:              (more about LIBLINEAR: https://github.com/cjlin1/liblinear)
    -c, --liblinearC        LIBLINEAR cost co-efficient, inverse of regularization strength, must be a positive float,
                            smaller values specify stronger regularization (default = 10.0)
    --eps, --liblinearEps   LIBLINEAR tolerance of termination criterion (default = 0.1)
    --solver, --liblinearSolver     LIBLINEAR solver (default for log loss = L2R_LR_DUAL, for l2 loss = L2R_L2LOSS_SVC_DUAL)
                                    Overrides default solver set by loss parameter.
                                    Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                                       L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
    --maxIter, --liblinearMaxIter   Maximum number of iterations for LIBLINEAR (default = 100)

    SGD/AdaGrad:
    -l, --lr, --eta         Step size (learning rate) for online optimizers (default = 1.0)
    --epochs                Number of training epochs for online optimizers (default = 1)
    --adagradEps            Defines starting step size for AdaGrad (default = 0.001)

    Tree (PLT and HSM):
    -a, --arity             Arity of tree nodes (default = 2)
    --maxLeaves             Maximum degree of pre-leaf nodes. (default = 100)
    --tree                  File with tree structure
    --treeType              Type of a tree to build if file with structure is not provided
                            tree types: hierarchicalKmeans, huffman, completeKaryInOrder, completeKaryRandom,
                                        balancedInOrder, balancedRandom, onlineComplete

    K-Means tree:
    --kmeansEps             Tolerance of termination criterion of the k-means clustering
                            used in hierarchical k-means tree building procedure (default = 0.001)
    --kmeansBalanced        Use balanced K-Means clustering (default = 1)

    Prediction:
    --topK                  Predict top-k labels (default = 5)
    --threshold             Predict labels with probability above the threshold (default = 0)
    --thresholds            Path to a file with threshold for each label, one threshold per line
    --labelsWeights         Path to a file with weight for each label, one weight per line
    --setUtility            Type of set-utility function for prediction using svbopFull, svbopHf, svbopMips models.
                            Set-utility functions: uP, uF1, uAlfa, uAlfaBeta, uDeltaGamma
                            See: https://arxiv.org/abs/1906.08129

    Set-Utility:
    --alpha
    --beta
    --delta
    --gamma

    Test:
    --measures              Evaluate test using set of measures (default = "p@1,p@3,p@5")
                            Measures: acc (accuracy), p (precision), r (recall), c (coverage), hl (hamming loos)
                                      p@k (precision at k), r@k (recall at k), c@k (coverage at k), s (prediction size)
```

See documentation for more details.


## References and acknowledgments

This library implements methods from following papers (see `experiments` directory for scripts to replicate the results):

- [Probabilistic Label Trees for Extreme Multi-label Classification](https://arxiv.org/abs/2009.11218)
- [Online probabilistic label trees](https://arxiv.org/abs/2007.04451)
- [Efficient Algorithms for Set-Valued Prediction in Multi-Class Classification](https://arxiv.org/abs/1906.08129)

Another implementation of PLT model is available in [extremeText](https://github.com/mwydmuch/extremeText) library, 
that implements approach described in this [NeurIPS paper](http://papers.nips.cc/paper/7872-a-no-regret-generalization-of-hierarchical-softmax-to-extreme-multi-label-classification).
