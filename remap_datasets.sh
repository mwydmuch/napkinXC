#!/usr/bin/env bash

set -e

DATASET="LSHTC1"
bash get_data.sh ${DATASET}
mv data/${DATASET} data/_${DATASET}
mkdir -p data/${DATASET}
python remap_labels.py data/_${DATASET}/${DATASET}.train data/${DATASET}/${DATASET}.train data/_${DATASET}/${DATASET}.test data/${DATASET}/${DATASET}.test

DATASET="Dmoz"
bash get_data.sh ${DATASET}
mv data/${DATASET} data/_${DATASET}
mkdir -p data/${DATASET}
python remap_labels.py data/_${DATASET}/${DATASET}.train data/${DATASET}/${DATASET}.train data/_${DATASET}/${DATASET}.test data/${DATASET}/${DATASET}.test
