# napkinXML

Extremely simple implementation of extreme multi-label classifier.

## TODO:
- Add balanced kmeans clustering
- Mix dense/sparse base classifiers for optimal speed/memory
- Parallelize example gathering
- Parallelize prediction
- Optimize types in LibLinear (it uses double everywhere, we could gain some speed by changing them to floats)
- Ensemble
- Better paths/arguments handling
- Add hashing

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
        note: -1 to use #cpus - 1, 0 to set #cpus
    --header        Input contains header (default = 1)
        header format: #lines #features #labels
    --hash          Size of hashing space (default = -1)
        note: -1 to disable

    Base classifier:
    -s, --solver    LibLinear solver (default = L2R_LR_DUAL)
        supported solvers: L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL
        see: https://github.com/cjlin1/liblinear
    -e, --eps       Stopping criteria (default = 0.1)
        see: https://github.com/cjlin1/liblinear
    --bias          Add bias term (default = 1)

    Tree:
    --arity         Arity of tree (default = 2)
    --tree          File with tree structure
    --treeType      Type of tree to build if file with structure is not provided
       tree types: completeInOrder, completeRandom, complete
```
