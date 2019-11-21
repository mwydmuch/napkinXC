# napkinXC [![Build Status](https://travis-ci.org/mwydmuch/napkiXC.svg?branch=master)](https://travis-ci.org/mwydmuch/napkiXC)

Extremely simple and fast library for extreme multi-class and multi-label classification.

Right now it implements:
- Binary Relevance (BR),
- One Versus Rest (OVR),
- OVA and BR with inference using Maximum Inner Product Search (MIPS),
- Hierarchical Softmax (HSM),
- Probabilistic Label Tree (PLT),
- Ensembles of tree based models,
- Top-k and set-valued prediction,
- LibLinear and SGD solvers for base classifiers,
- Online prediction,
- Huffman, complete and balanced tree structures,
- Hierarchical balanced k-means clustering for tree building,
- Loading custom tree structures.

Please note that this library is still under development and serves as a base for experiments.
Some of the features may not be listed in options below.

This repository contains code for [arXiv paper](https://arxiv.org/abs/1906.08129) about set-valued prediction in multi-class classifcation.

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
Usage: nxml <command> <args>

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
    -t, --threads       Number of threads used for training and testing (default = -1)
                        Note: -1 to use #cpus - 1, 0 to use #cpus
    --header            Input contains header (default = 1)
                        Header format for libsvm: #lines #features #labels
    --hash              Size of features space (default = 0)
                        Note: 0 to disable hashing
    --featuresThreshold Prune features below given threshold (default = 0.04)             
    --seed              Seed

    Base classifiers:
    --optimizer         Use LibLiner or online optimizers (default = libliner)
                        Optimizers: liblinear, sgd, adagrad
    --bias              Add bias term (default = 1)
    --labelsWeights     Increase the weight of minority labels in base classifiers (default = 1)
    --weightsThreshold  Prune weights below given threshold (default = 0.1)

    LibLinear:
    -s, --solver        LibLinear solver (default = L2R_LR_DUAL)
                        Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                           L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
                        See: https://github.com/cjlin1/liblinear
    -c, -C, --cost      Inverse of regularization strength. Must be a positive float.
                        Like in support vector machines, smaller values specify stronger
                        regularization. (default = 10.0)
                        Note: -1 to automatically find best value for each node.
    -e, --eps           Stopping criteria (default = 0.1)
                        See: https://github.com/cjlin1/liblinear

    SGD/AdaGrad:
    -e, --eta           Step size of SGD
    --epochs            Number of epochs of SGD

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
                        Measures: acc, p, r, c, p@k, r@k, c@k, s
                        
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

