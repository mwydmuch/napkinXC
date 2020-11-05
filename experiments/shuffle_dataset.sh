#!/usr/bin/env bash

DATASET="$1"
SEED=$2

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
DATASET_DIR="${SCRIPT_DIR}/../data/${DATASET}"
DATASET_PREFIX="${SCRIPT_DIR}/../data/${DATASET}/${DATASET}"
NEW_DATASET_DIR="${DATASET_DIR}_--seed_${SEED}"
NEW_DATASET_PREFIX="${NEW_DATASET_DIR}/${DATASET}_--seed_${SEED}"

get_random_source () {
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

if [ ! -e ${NEW_DATASET_DIR} ]; then
    mkdir -p ${NEW_DATASET_DIR}
    cp ${DATASET_PREFIX}_test.txt ${NEW_DATASET_PREFIX}_test.txt

    # Remove header and shuffle lines to temporary file
    tail -n +2 ${DATASET_PREFIX}_train.txt | shuf --random-source=<(get_random_source ${SEED}) > ${NEW_DATASET_PREFIX}_tmp.txt

    # Create train file
    head -n 1 ${DATASET_PREFIX}_train.txt > ${NEW_DATASET_PREFIX}_train.txt
    cat ${NEW_DATASET_PREFIX}_tmp.txt >> ${NEW_DATASET_PREFIX}_train.txt

    # Remove temporary file
    rm ${NEW_DATASET_PREFIX}_tmp.txt
fi
