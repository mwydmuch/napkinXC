Using C++ executable
====================

napkinXC can also be built and used as an executable that can be used to train and evaluate models and make a prediction.


Building
--------

To build napkinXC, first clone the project repository and run the following commands in the root directory of the project.
It requires modern C++17 compiler, CMake and Git installed.
Set CXX and CC environmental variables before running ``cmake`` command if you want to build with the specific C++ compiler.

.. code:: sh

    cmake .
    make


``-B`` options can be passed to CMake command to specify other build directory.
After successful compilation, ``nxc`` executable should appear in the root or specified build directory.


LIBSVM data format
------------------

napkinXC supports multi-label svmlight/libsvm like-format (less strict)
and format of datasets from `The Extreme Classification Repository <https://manikvarma.github.io/downloads/XC/XMLRepository.html>`_,
which has an additional header line with a number of data points, features, and labels.

The format is text-based. Each line contains an instance and is ended by a ``\n`` character.

.. code::

    <label>,<label>,... <feature>(:<value>) <feature>(:<value>) ...

``<label>`` and ``<feature>`` are indexes that should be positive integers.
Unlike to normal svmlight/libsvm format, labels and features do not have to be sorted in ascending order.
The ``:<value>`` can be omitted after ``<feature>``, to assume value = 1.

Usage
-----

``nxc`` executable needs command, i.e. train, test, predict as a first argument.
``-i``/``--input`` and ``-o``/``--output`` arguments needs to be always provided.

.. code:: sh

    nxc <command> -i <path to dataset> -o <path to model directory> <args> ...


Command line options
--------------------

.. code::

    Usage: nxc <command> <args>

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
        -m, --model             Model type (default = plt):
                                Models: ovr, br, hsm, plt, oplt, svbopFull, svbopHf, brMips, svbopMips
        --ensemble              Number of models in ensemble (default = 1)
        -t, --threads           Number of threads to use (default = 0)
                                Note: -1 to use #cpus - 1, 0 to use #cpus
        --hash                  Size of features space (default = 0)
                                Note: 0 to disable hashing
        --featuresThreshold     Prune features below given threshold (default = 0.0)
        --seed                  Seed (default = system time)
        --verbose               Verbose level (default = 2)

        Base classifiers:
        --optimizer             Optimizer used for training binary classifiers (default = liblinear)
                                Optimizers: liblinear, sgd, adagrad, fobos
        --bias                  Value of the bias features (default = 1)
        --weightsThreshold      Threshold value for pruning models weights (default = 0.1)

        LIBLINEAR:              (more about LIBLINEAR: https://github.com/cjlin1/liblinear)
        -s, --liblinearSolver   LIBLINEAR solver (default for log loss = L2R_LR_DUAL, for l2 loss = L2R_L2LOSS_SVC_DUAL)
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
        --threshold             Predict labels with probability above the threshold (default = 0)
        --thresholds            Path to a file with threshold for each label

        Test:
        --measures              Evaluate test using set of measures (default = "p@1,r@1,c@1,p@3,r@3,c@3,p@5,r@5,c@5")
                                Measures: acc (accuracy), p (precision), r (recall), c (coverage), hl (hamming loos)
                                          p@k (precision at k), r@k (recall at k), c@k (coverage at k), s (prediction size)
