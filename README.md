# napkinXML

Extremely simple and fast extreme multi-label classifier based on Probabilistic Label Tree (PLT) algorithm.

## Notes
Feel free to work on this repo, if you like.
I'm (Marek) currently working on balanced kmeans clustering.

## TODOs
- Balanced kmeans clustering
- Parallel example gathering
- Optimise types in LibLinear (it uses double, we could gain some speed by changing to floats)
- Ensemble of trees (reuse code from fastText version)
- Better paths handling and checking (<dir>/<dir2> works, <dir>/<dir2>/ doesn't)
- Feature hashing
- Storing the model in one large file instead of separate small files

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
    -a, --arity     Arity of a tree (default = 2)
    --tree          File with tree structure
    --treeType      Type of a tree to build if file with structure is not provided
                    Tree types: completeInOrder, completeRandom, complete
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
