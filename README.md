# napkinXC [![Build Status](https://travis-ci.org/mwydmuch/napkinXC.svg?branch=master)](https://travis-ci.org/mwydmuch/napkinXC)

Extremely simple and fast library for extreme multi-class and multi-label classification.

Right now it implements:
- Binary Relevance (BR),
- One Versus Rest (OVR),
- OVA and BR with inference using Maximum Inner Product Search (MIPS),
- Hierarchical Softmax (HSM),
- Probabilistic Label Tree (PLT),
- extremeText (XT),
- Ensembles of tree based models,
- Top-k and set-valued prediction,
- LibLinear, SGD and AdaGrad solvers for base classifiers,
- Online prediction,
- Huffman, complete and balanced tree structures,
- Hierarchical k-means clustering for tree building,
- Loading custom tree structures.

Please note that this library is still under development and serves as a base for experiments.
Features may change or break and some of the options may not be listed below.

This library implements methods flow following papers:

```
@article{DBLP:journals/corr/abs-1906-08129,
  author    = {Thomas Mortier and
               Marek Wydmuch and
               Eyke H{\"{u}}llermeier and
               Krzysztof Dembczynski and
               Willem Waegeman},
  title     = {Efficient Algorithms for Set-Valued Prediction in Multi-Class Classification},
  journal   = {CoRR},
  volume    = {abs/1906.08129},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.08129},
  archivePrefix = {arXiv},
  eprint    = {1906.08129},
  timestamp = {Mon, 24 Jun 2019 17:28:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1906-08129.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

This repository contains code for this [arXiv paper](https://arxiv.org/abs/1906.08129) about set-valued prediction in multi-class classification.

Another implementation of PLT model is available in [extremeText](https://github.com/mwydmuch/extremeText) library, that implements approach described in this [NeurIPS paper](http://papers.nips.cc/paper/7872-a-no-regret-generalization-of-hierarchical-softmax-to-extreme-multi-label-classification).

## Build
```
cmake -DCMAKE_BUILD_TYPE=Release .
make -j
```

To build with MIPS-based models library requires [Non-Metric Space Library (NMSLIB)](https://github.com/nmslib/nmslib.git):

To install NMSLIB:
```
git clone https://github.com/nmslib/nmslib.git
cd nmsilb/similarity_search
cmake -DCMAKE_BUILD_TYPE=Release .
make -j
make install
```

To build with MIPS extension:
```
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_MIPS_EXT=1 .
make -j
```

## Options

```
Usage: nxc <command> <args>

Commands:
    train
    test
    predict

Args:
    General:
    -i, --input         Input dataset
    -o, --output        Output (model) dir
    -m, --model         Model type (default = plt):
                        Models: ovr, br, hsm, plt, oplt, ubop, rbop,
                                ubopHsm, brMips, ubopMips
    --ensemble          Ensemble of models (default = 0)
    -d, --dataFormat    Type of data format (default = libsvm):
                        Supported data formats: libsvm
    -t, --threads       Number of threads used for training and testing (default = 0)
                        Note: -1 to use #cpus - 1, 0 to use #cpus
    --header            Input contains header (default = 1)
                        Header format for libsvm: #lines #features #labels
    --hash              Size of features space (default = 0)
                        Note: 0 to disable hashing
    --featuresThreshold Prune features belowe given threshold (default = 0.0)
    --seed              Seed

    Base classifiers:
    --optimizer         Use LibLiner or online optimizers (default = libliner)
                        Optimizers: liblinear, sgd, adagrad, fobos
    --bias              Add bias term (default = 1)
    --inbalanceLabelsWeighting     Increase the weight of minority labels in base classifiers (default = 1)
    --weightsThreshold  Prune weights belowe given threshold (default = 0.1)

    LibLinear:
    -s, --solver        LibLinear solver (default = L2R_LR_DUAL)
                        Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                           L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
                        See: https://github.com/cjlin1/liblinear
    -c, -C, --cost      Inverse of regularization strength. Must be a positive float.
                        Like in support vector machines, smaller values specify stronger
                        regularization. (default = 1.0)
                        Note: -1 to automatically find best value for each node.
    -e, --eps           Stopping criteria (default = 0.1)
                        See: https://github.com/cjlin1/liblinear

    SGD/AdaGrad:
    -l, --lr, --eta     Step size (learning rate) of SGD/AdaGrad (default = 1.0)
    --epochs            Number of epochs of SGD/AdaGrad (default = 3)
    --adagradEps        AdaGrad epsilon (default = 0.001)

    Tree:
    -a, --arity         Arity of a tree (default = 2)
    --maxLeaves         Maximum number of leaves (labels) in one internal node. (default = 100)
    --tree              File with tree structure
    --treeType          Type of a tree to build if file with structure is not provided
                        Tree types: hierarchicalKMeans, huffman, completeInOrder, completeRandom,
                                    balancedInOrder, balancedRandom, onlineComplete, onlineBalanced,
                                    onlineRandom

    K-Means tree:
    --kMeansEps         Stopping criteria for K-Means clustering (default = 0.001)
    --kMeansBalanced    Use balanced K-Means clustering (default = 1)

    Prediction:
    --topK              Predict top k elements (default = 5)
    --setUtility        Type of set-utility function for prediction using ubop, rbop, ubopHsm, ubopMips models.
                        Set-utility functions: uP, uF1, uAlfa, uAlfaBeta, uDeltaGamma
                        See: https://arxiv.org/abs/1906.08129

    Set-Utility:
    --alfa
    --beta
    --delta
    --gamma

    Test:
    --measures          Evaluate test using set of measures (default = "p@1,r@1,c@1,p@3,r@3,c@3,p@5,r@5,c@5")
                        Measures: acc (accuracy), p (precision), r (recall), c (coverage),
                                  p@k (precision at k), r@k (recall at k), c@k (coverage at k), s (prediction size)
```

## Test script
```
Usage test.sh <dataset> <optional nxml train args>

Datasets:
    Multi-label:
        amazonCat
        amazonCat-14K
        amazon
        amazon-3M
        deliciousLarge
        eurlex
        wiki10
        wikiLSHTC
        WikipediaLarge-500K
```

## TODO
- Proper logging with verbose options
- Python bindings with support for SciPy types

## Acknowledgments
napkinXC uses the following libraries:

- LIBLINEAR: https://github.com/cjlin1/liblinear
- ThreadPool: https://github.com/progschj/ThreadPool
- Robin Hood Hashmap: https://github.com/martinus/robin-hood-hashing
- Non-Metric Space Library: https://github.com/nmslib/nmslib.git

