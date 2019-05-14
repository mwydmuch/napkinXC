#!/usr/bin/env bash

set -e

REMAP () {
    DATASET=$1
    DATASET_DIR=data/${DATASET}
    echo "Remapping ${DATASET} ..."
    if [ ! -e $DATASET_DIR ]; then
        bash get_data.sh ${DATASET}
    fi
    python remap_dataset.py ${DATASET_DIR}/${DATASET}.train ${DATASET_DIR}/${DATASET}.test

}

REMAP "LSHTC1"
python LSHTC1_convert_hier.py data/LSHTC1

python Dmoz_merge_test_files.py data/Dmoz
REMAP "Dmoz"
python Dmoz_convert_hier.py data/Dmoz
