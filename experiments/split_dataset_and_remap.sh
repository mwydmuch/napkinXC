#!/usr/bin/env bash

DATASET="$1"
VALID_SIZE=$2
SEED=$3

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
DATASET_DIR="${SCRIPT_DIR}/../data/${DATASET}"
DATASET_PREFIX="${SCRIPT_DIR}/../data/${DATASET}/${DATASET}"
NEW_DATASET_DIR="${DATASET_DIR}_--split_${VALID_SIZE}_--seed_${SEED}"
NEW_DATASET_PREFIX="${NEW_DATASET_DIR}/${DATASET}_--split_${VALID_SIZE}_--seed_${SEED}"

get_random_source () {
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

if [ ! -e ${NEW_DATASET_DIR} ]; then
    mkdir -p ${NEW_DATASET_DIR}
    cp ${DATASET_PREFIX}_test.txt ${NEW_DATASET_PREFIX}_test.txt

    # Remove header and shuffle lines to temporary file
    tail -n +2 ${DATASET_PREFIX}_train.txt | shuf --random-source=<(get_random_source ${SEED}) > ${NEW_DATASET_PREFIX}_tmp.txt

    # Calculate size of valid dataset
    ALL_SIZE=$(wc -l ${NEW_DATASET_PREFIX}_tmp.txt | xargs | cut -d " " -f1)
    VALID_SIZE=$[ ${ALL_SIZE} * ${VALID_SIZE} / 100 ]
    TRAIN_SIZE=$[ ${ALL_SIZE} - ${VALID_SIZE} ]

    # Create validation file
    head -n ${VALID_SIZE} ${NEW_DATASET_PREFIX}_tmp.txt > ${NEW_DATASET_PREFIX}_valid.txt

    # Create train file
    tail -n +$[ ${VALID_SIZE} + 1 ] ${NEW_DATASET_PREFIX}_tmp.txt > ${NEW_DATASET_PREFIX}_train.txt

    # Remove header from test file
    tail -n +2 ${DATASET_PREFIX}_test.txt > ${NEW_DATASET_PREFIX}_test.txt

    # Remove temporary file
    rm ${NEW_DATASET_PREFIX}_tmp.txt

    # Remap files
    python3 ${SCRIPT_DIR}/remap_libsvm.py ${NEW_DATASET_PREFIX}_train.txt ${NEW_DATASET_PREFIX}_valid.txt ${NEW_DATASET_PREFIX}_test.txt
    rm ${NEW_DATASET_PREFIX}_train.txt ${NEW_DATASET_PREFIX}_valid.txt ${NEW_DATASET_PREFIX}_test.txt
fi
