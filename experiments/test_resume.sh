#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
ROOT_DIR=${SCRIPT_DIR}/..

DATASET_NAME=$1
MODEL_DIR=models
RESULTS_DIR=results

# If there are exactly 4 arguments and 3 starts with nxc parameter (-)
if [[ $# -gt 3 ]] && [[ $2 == -* ]] && [[ $3 == -* ]] && [[ $4 == -* ]]; then
    TRAIN_ARGS=$2
    TRAIN_RESUME_ARGS=$3
    TEST_ARGS=$4
    if [[ $# -gt 4 ]]; then
        MODEL_DIR=$5
    fi
    if [[ $# -gt 5 ]]; then
        RESULTS_DIR=$6
    fi
else
    shift
    TRAIN_ARGS="$@"
    TRAIN_RESUME_ARGS=""
    TEST_ARGS=""
fi

TRAIN_CONFIG=${DATASET_NAME}_$(echo "${TRAIN_ARGS}" | tr " /" "__")_$(echo "${TRAIN_RESUME_ARGS}" | tr " /" "__")
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
    TRAIN_RESUME_FILE="${DATASET_FILE}.valid.remapped"
    TEST_FILE="${DATASET_FILE}.test.remapped"
elif [[ -e "${DATASET_FILE}_train.txt.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt.remapped"
    TRAIN_RESUME_FILE="${DATASET_FILE}_valid.txt.remapped"
    TEST_FILE="${DATASET_FILE}_test.txt.remapped"
elif [[ -e "${DATASET_FILE}_train.txt" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt"
    TRAIN_RESUME_FILE="${DATASET_FILE}_valid.txt"
    TEST_FILE="${DATASET_FILE}_test.txt"
elif [[ -e "${DATASET_FILE}.train" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train"
    TRAIN_RESUME_FILE="${DATASET_FILE}.valid"
    TEST_FILE="${DATASET_FILE}.test"
elif [[ -e "${DATASET_FILE}_train.svm" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.svm"
    TRAIN_RESUME_FILE="${DATASET_FILE}_valid.svm"
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
    (time ${ROOT_DIR}/nxc train --saveGrads 1 -i $TRAIN_FILE -o $MODEL $TRAIN_ARGS | tee $TRAIN_RESULT_FILE)
    echo
    echo "Train date: $(date)" | tee -a $TRAIN_RESULT_FILE
    rm -f $TRAIN_LOCK_FILE
fi

# Train resume model
TRAIN_RESUME_RESULT_FILE=${MODEL}/train_resume_results
TRAIN_RESUME_LOCK_FILE=${MODEL}/.train_resume_lock
if [[ ! -e $TRAIN_RESUME_RESULT_FILE ]] || [[ -e $TRAIN_RESUME_LOCK_FILE ]]; then
    touch $TRAIN_RESUME_LOCK_FILE
    (time ${ROOT_DIR}/nxc train --resume 1 -i $TRAIN_RESUME_FILE -o $MODEL $TRAIN_RESUME_ARGS | tee $TRAIN_RESUME_RESULT_FILE)
    echo
    echo "Train date: $(date)" | tee -a $TRAIN_RESUME_RESULT_FILE
    rm -f $TRAIN_RESUME_LOCK_FILE
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
    if [ -e $TRAIN_RESUME_RESULT_FILE ]; then
        cat $TRAIN_RESUME_RESULT_FILE > $TEST_RESULT_FILE
    fi
    (time ${ROOT_DIR}/nxc test -i $TEST_FILE -o $MODEL $TEST_ARGS | tee -a $TEST_RESULT_FILE)

    echo
    echo "Model file size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)" | tee -a $TEST_RESULT_FILE
    echo "Model file size (K): $(du -c ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+' -o)" | tee -a $TEST_RESULT_FILE
    echo "Test date: $(date)" | tee -a $TEST_RESULT_FILE
    rm -rf $TEST_LOCK_FILE
else
    cat $TEST_RESULT_FILE
fi
