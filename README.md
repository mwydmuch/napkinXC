# napkinXML

Extremely simple implementation of extreme multi-label classifier.

## TODO:
- Add balanced kmeans clustering
- Parallelize example gathering
- Optimize types in LibLinear (it uses double, we could gain some speed by changing to floats)
- Ensemble of trees
- Better paths/arguments handling
- Add feature hashing

## Build
```
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
    -i, --input     Input dataset in LibSvm format
    -m, --model     Model's dir
    -t, --threads   Number of threads used for training and testing (default = -1)
                    Note: -1 to use #cpus - 1, 0 to use #cpus
    --header        Input contains header (default = 1)
                    Header format: #lines #features #labels
                    Note: XML repo datasets have headers
    --hash          Size of hashing space (default = -1)
                    Note: -1 to disable

    Base classifier:
    -s, --solver    LibLinear solver (default = L2R_LR_DUAL)
                    Supported solvers: L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC,
                    L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL
                    See: https://github.com/cjlin1/liblinear
    -e, --eps       Stopping criteria (default = 0.1)
                    See: https://github.com/cjlin1/liblinear
    --bias          Add bias term (default = 1)

    Tree:
    --arity         Arity of a tree (default = 2)
    --tree          File with tree structure
    --treeType      Type of a tree to build if file with structure is not provided
                    Tree types: completeInOrder, completeRandom, complete
```
