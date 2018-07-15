# napkinXML

Extremely simple and fast extreme multi-label classifier based on Probabilistic Label Tree (PLT) algorithm.

Right now it implements:
- LibLinear and SGD solvers for tree nodes,
- Online prediction,
- Basic complete, Huffman and balanced tree structures,
- Hierarchical balanced k-means clustering for tree building,
- Loading custom tree structures,
- KNN for super-labels (last level nodes with a large number of leaves)

Please note that this library is still under development and serves as a base for experiments that aim to improve PLT algorithm.
Some of the features may not be listed in options below.

## Build
```
rm -f CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=Release
make
```

## Options

```
Usage: nxml <command> <args>

Commands:
    train
    test

Args:
    General:
    -i, --input         Input dataset in LibSvm format
    -m, --model         Model's dir
    -t, --threads       Number of threads used for training and testing (default = -1)
                        Note: -1 to use #cpus - 1, 0 to use #cpus
    --header            Input contains header (default = 1)
                        Header format: #lines #features #labels
    --hash              Size of hashing space (default = 0)
                        Note: 0 to disable
    --seed              Model's seed

    Base classifiers:
    --optimizer         Use LibLiner or SGD (default = libliner)
                        Optimizers: liblinear, sgd
    --bias              Add bias term (default = 1)
    --labelsWeights     Increase the weight of minority labels in base classifiers (default = 1)

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

    SGD:
    -e, --eta           Step size of SGD
    --iter              Number of epochs of SGD

    Tree:
    -a, --arity         Arity of a tree (default = 2)
    --maxLeaves         Maximum number of leaves (labels) in one internal node. (default = 100)
    --tree              File with tree structure
    --treeType          Type of a tree to build if file with structure is not provided
                        Tree types: hierarchicalKMeans, huffman, completeInOrder, completeRandom,
                                    balancedInOrder, balancedRandom,

    K-Means tree:
    --kMeansEps         Stopping criteria for K-Means clustering (default = 0.001)
    --kMeansBalanced    Use balanced K-Means clustering (default = 1)

    Random projection:
    --projectDim        Number or random direction

    K-NNs:
    --kNN               Number of nearest neighbors used for prediction

```

## Tests
```
Usage test.sh <dataset> <optional nxml train args>

Datasets:
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
