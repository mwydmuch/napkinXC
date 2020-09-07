# napkinXC [![Build Status](https://travis-ci.org/mwydmuch/napkinXC.svg?branch=master)](https://travis-ci.org/mwydmuch/napkinXC)

napkinXC is an extremely simple and fast library for extreme multi-class and multi-label classification.
It allows to train a classifier for very large datasets in few lines of code with minimal resources.

Right now, napkinXC implements the following features both in Python and C++:
- Probabilistic Label Trees (PLT) and Online Probabilistic Label Trees (OPLT),
- Hierarchical softmax (HSM),
- Binary Relevance (BR),
- One Versus Rest (OVR),
- fast online prediction of top-k labels or labels above the given threshold,
- hierarchical k-means clustering for tree building and other tree building methods,
- support for predefined hierarchies,
- LIBLINEAR, SGD, and AdaGrad solvers for base classifiers,
- efficient ensembles tree-based model,
- helpers to download and load data from [XML Repository](http://manikvarma.org/downloads/XC/XMLRepository.html),
- helpers to measure performance.

Please note that this library is still under development and also serves as a base for experiments. 
Some of the experimental features may not be documented. 

The napkinXC is distributed under MIT license. 
All contributions to the project are welcome!


## Roadmap

Coming soon:
- OPLT available in Python
- Possibility to use any type of binary classifier from Python
- Improved dataset loading from Python
- More datasets from XML Repository


## Python quick start

Python version of napkinXC can be easly installed from PyPy repository:
```
pip install napkinxc
```

Minimal example of usage:
```
from napkinxc.models import PLT
from napkinxc.measures import precision_at_k
from napkinxc.datasets import load_dataset

X_train, Y_train = load_dataset("eurlex-4k", "train")
X_test, Y_test = load_dataset("eurlex-4k", "test")
plt = PLT("eurlex-model")
plt.fit(X_train, Y_train)
Y_pred = plt.predict(X_test, top_k=1)
print(precision_at_k(Y_test, Y_pred, k=1))
```

More examples can be found under `python/examples` directory.


## Building executable

napkinXC can be also build as executable using:

```
cmake .
make -j
```


## Command line options

```
Usage: nxc <command> <args>

Commands:
    train                   Train model on given input data
    test                    Test model on given input data
    predict                 Predict for given data
    ofo                     Use online f-measure optimalization
    version                 Print napkinXC version
    help                    Print help

Args:
    General:
    -i, --input             Input dataset
    -o, --output            Output (model) dir
    -m, --model             Model type (default = plt):
                            Models: ovr, br, hsm, plt, oplt, ubop, ubopHsm, brMips, ubopMips
    --ensemble              Number of models in ensemble (default = 1)
    -d, --dataFormat        Type of data format (default = libsvm),
                            Supported data formats: libsvm
    -t, --threads           Number of threads to use (default = 0)
                            Note: -1 to use #cpus - 1, 0 to use #cpus
    --header                Input contains header (default = 1)
                            Header format for libsvm: #lines #features #labels
    --hash                  Size of features space (default = 0)
                            Note: 0 to disable hashing
    --featuresThreshold     Prune features below given threshold (default = 0.0)
    --seed                  Seed (default = system time)
    --verbose               Verbose level (default = 2)

    Base classifiers:
    --optimizer             Optimizer used for training binary classifiers (default = libliner)
                            Optimizers: liblinear, sgd, adagrad, fobos
    --bias                  Value of the bias features (default = 1)
    --inbalanceLabelsWeighting     Increase the weight of minority labels in base classifiers (default = 1)
    --weightsThreshold      Threshold value for pruning models weights (default = 0.1)

    LIBLINEAR:              (more aobut LIBLINEAR: https://github.com/cjlin1/liblinear)
    -s, --solver            LIBLINEAR solver (default for log loss = L2R_LR_DUAL, for l2 loss = L2R_L2LOSS_SVC_DUAL)
                            Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                               L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
    -c, --liblinearC        LIBLINEAR cost co-efficient, inverse of regularization strength, must be a positive float,
                            smaller values specify stronger regularization (default = 10.0)
    --eps, --liblinearEps   LIBLINEAR tolerance of termination criterion (default = 0.1)

    SGD/AdaGrad:
    -l, --lr, --eta         Step size (learning rate) for online optimizers (default = 1.0)
    --epochs                Number of training epochs for online optimizers (default = 1)
    --adagradEps            Defines starting step size for AdaGrad (default = 0.001)

    Tree:
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
    --threshold             Predict labels with probability above the threshold, defaults to 0
    --setUtility            Type of set-utility function for prediction using ubop, rbop, ubopHsm, ubopMips models.
                            Set-utility functions: uP, uF1, uAlfa, uAlfaBeta, uDeltaGamma
                            See: https://arxiv.org/abs/1906.08129

    Set-Utility:
    --alfa
    --beta
    --delta
    --gamma

    Test:
    --measures              Evaluate test using set of measures (default = "p@1,r@1,c@1,p@3,r@3,c@3,p@5,r@5,c@5")
                            Measures: acc (accuracy), p (precision), r (recall), c (coverage), hl (hamming loos)
                                      p@k (precision at k), r@k (recall at k), c@k (coverage at k), s (prediction size)
```


## References and acknowledgments

This library implements methods from following papers:

- [Online Probabilistic Label Trees](https://arxiv.org/abs/1906.08129)

- [Efficient Algorithms for Set-Valued Prediction in Multi-Class Classification](https://arxiv.org/abs/1906.08129)

Another implementation of PLT model is available in [extremeText](https://github.com/mwydmuch/extremeText) library, 
that implements approach described in this [NeurIPS paper](http://papers.nips.cc/paper/7872-a-no-regret-generalization-of-hierarchical-softmax-to-extreme-multi-label-classification).
