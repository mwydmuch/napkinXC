#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
ROOT_DIR=${SCRIPT_DIR}/..

DATASET_NAME=$1
MODEL_DIR=models
PRED_DIR=predictions

# If there are exactly 3 arguments and 2 starts with nxc parameter (-)
if [[ $# -gt 2 ]] && [[ $2 == -* ]] && [[ $3 == -* ]]; then
    TRAIN_ARGS=$2
    PRED_ARGS=$3
    if [[ $# -gt 3 ]]; then
        MODEL_DIR=$4
    fi
    if [[ $# -gt 4 ]]; then
        PRED_DIR=$5
    fi
else
    shift
    TRAIN_ARGS="$@"
    PRED_ARGS=""
fi

TRAIN_CONFIG=${DATASET_NAME}_$(echo "${TRAIN_ARGS}" | tr " /" "__")
PRED_CONFIG=${TRAIN_CONFIG}_$(echo "${PRED_ARGS}" | tr " /" "__")

MODEL=${MODEL_DIR}/${TRAIN_CONFIG}
DATASET_DIR=data/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}

# Download dataset
if [[ ! -e $DATASET_DIR ]]; then
    bash ${SCRIPT_DIR}/get_dataset.sh $DATASET_NAME
fi

# Find train / test file
if [[ -e "${DATASET_FILE}.train.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train.remapped"
    PRED_FILE="${DATASET_FILE}.test.remapped"
elif [[ -e "${DATASET_FILE}_train.txt.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt.remapped"
    PRED_FILE="${DATASET_FILE}_test.txt.remapped"
elif [[ -e "${DATASET_FILE}_train.txt" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt"
    PRED_FILE="${DATASET_FILE}_test.txt"
elif [[ -e "${DATASET_FILE}.train" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train"
    PRED_FILE="${DATASET_FILE}.test"
elif [[ -e "${DATASET_FILE}_train.svm" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.svm"
    PRED_FILE="${DATASET_FILE}_test.svm"
fi

# Build nxc
if [[ ! -e ${ROOT_DIR}/nxc ]]; then
    cd ${ROOT_DIR}
    rm -f CMakeCache.txt
    cmake -DCMAKE_BUILD_TYPE=Release .
    make -j
    cd ${ROOT_DIR}/experiments
fi

# Train model
TRAIN_RESULT_FILE=${MODEL}/train_results
TRAIN_LOCK_FILE=${MODEL}/.train_lock
if [[ ! -e $MODEL ]] || [[ -e $TRAIN_LOCK_FILE ]]; then
    mkdir -p $MODEL
    touch $TRAIN_LOCK_FILE
    (time ${ROOT_DIR}/nxc train -i $TRAIN_FILE -o $MODEL $TRAIN_ARGS | tee $TRAIN_RESULT_FILE)
    echo
    echo "Train date: $(date)" | tee -a $TRAIN_RESULT_FILE
    rm -f $TRAIN_LOCK_FILE
fi

# Predict
PRED_RESULT_FILE=${PRED_DIR}/${PRED_CONFIG}
PRED_LOCK_FILE=${PRED_DIR}/.test_lock_${PRED_CONFIG}
if [[ ! -e $PRED_RESULT_FILE ]] || [[ -e $PRED_LOCK_FILE ]]; then
    mkdir -p $PRED_DIR
    touch $PRED_LOCK_FILE
    ${ROOT_DIR}/nxc predict -i $PRED_FILE -o $MODEL $PRED_ARGS > ${PRED_RESULT_FILE}
    rm -rf $PRED_LOCK_FILE
fi
