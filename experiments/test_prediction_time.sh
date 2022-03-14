#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
ROOT_DIR=${SCRIPT_DIR}/..

DATASET_NAME=$1
MODEL_DIR=models
RESULTS_DIR=results

# If there are exactly 3 arguments and 2 starts with nxc parameter (-)
if [[ $# -gt 2 ]] && [[ $2 == -* ]] && [[ $3 == -* ]]; then
    TRAIN_ARGS=$2
    TEST_ARGS=$3
    if [[ $# -gt 3 ]]; then
        MODEL_DIR=$4
    fi
    if [[ $# -gt 4 ]]; then
        RESULTS_DIR=$5
    fi
else
    shift
    TRAIN_ARGS="$@"
    TEST_ARGS=""
fi

TRAIN_CONFIG=${DATASET_NAME}_$(echo "${TRAIN_ARGS}" | tr " /" "__")
TEST_CONFIG=${TRAIN_CONFIG}_$(echo "${TEST_ARGS}" | tr " /" "__")

MODEL=${MODEL_DIR}/${TRAIN_CONFIG}
DATASET_DIR=data/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}

# Download dataset
if [[ ! -e $DATASET_DIR ]]; then
    python3 ${SCRIPT_DIR}/get_dataset-v1.py $DATASET_NAME
fi

# Find train / test file
if [[ -e "${DATASET_FILE}.train.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train.remapped"
    TEST_FILE="${DATASET_FILE}.test.remapped"
elif [[ -e "${DATASET_FILE}_train.txt" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt"
    TEST_FILE="${DATASET_FILE}_test.txt"
elif [[ -e "${DATASET_FILE}.train" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train"
    TEST_FILE="${DATASET_FILE}.test"
elif [[ -e "${DATASET_FILE}_train.svm" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.svm"
    TEST_FILE="${DATASET_FILE}_test.svm"
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

## Test model
TEST_RESULT_FILE=${RESULTS_DIR}/${TEST_CONFIG}
TEST_LOCK_FILE=${RESULTS_DIR}/.test_lock_${TEST_CONFIG}
if [[ ! -e $TEST_RESULT_FILE ]] || [[ -e $TEST_LOCK_FILE ]]; then
    mkdir -p $RESULTS_DIR
    touch $TEST_LOCK_FILE
    if [ -e $TRAIN_RESULT_FILE ]; then
        cat $TRAIN_RESULT_FILE > $TEST_RESULT_FILE
    fi
    (time ${ROOT_DIR}/nxc testPredictionTime -i $TEST_FILE -o $MODEL $TEST_ARGS | tee -a $TEST_RESULT_FILE)

    echo
    echo "Test date: $(date)" | tee -a $TEST_RESULT_FILE
    rm -rf $TEST_LOCK_FILE
else
    cat $TEST_RESULT_FILE
fi
