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
    python3 ${SCRIPT_DIR}/get_dataset.py $DATASET_NAME bow-v1
fi

# Find train / test file
if [[ -e "${DATASET_FILE}.train.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train.remapped"
    TEST_FILE="${DATASET_FILE}.test.remapped"
elif [[ -e "${DATASET_FILE}_train.txt.remapped" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt.remapped"
    TEST_FILE="${DATASET_FILE}_test.txt.remapped"
elif [[ -e "${DATASET_FILE}_train.txt" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt"
    TEST_FILE="${DATASET_FILE}_test.txt"
elif [[ -e "${DATASET_FILE}.train" ]]; then
    TRAIN_FILE="${DATASET_FILE}.train"
    TEST_FILE="${DATASET_FILE}.test"
elif [[ -e "${DATASET_FILE}_train.libsvm" ]]; then
    TRAIN_FILE="${DATASET_FILE}_train.libsvm"
    TEST_FILE="${DATASET_FILE}_test.libsvm"
fi

# Build nxc
if [[ ! -e ${ROOT_DIR}/nxc ]]; then
    cd ${ROOT_DIR}
    rm -f CMakeCache.txt
    cmake -DCMAKE_BUILD_TYPE=Release .
    make -j
    cd ${ROOT_DIR}/experiments
fi

# Calculate inverse propensity
INV_PS_FILE="${DATASET_FILE}.inv_ps"
if [[ ! -e $INV_PS_FILE ]]; then
    python3 ${SCRIPT_DIR}/calculate_Jain_et_al_inv_ps.py $TRAIN_FILE $INV_PS_FILE
fi

# Calculate inverse priors
# INV_PRIORS_FILE="${DATASET_FILE}.inv_priors"
# if [[ ! -e $INV_PRIORS_FILE ]]; then
#     python3 ${SCRIPT_DIR}/calculate_inv_priors.py $TRAIN_FILE $INV_PRIORS_FILE
# fi

# Train model
TRAIN_RESULT_FILE=${MODEL}/train_results
TRAIN_LOCK_FILE=${MODEL}/.train_lock
if [[ ! -e $MODEL ]] || [[ -e $TRAIN_LOCK_FILE ]]; then
    mkdir -p $MODEL
    touch $TRAIN_LOCK_FILE

    if [[ $TRAIN_ARGS == *"--labelsWeights"* ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --labelsWeights ${INV_PS_FILE}"
    fi

    ${ROOT_DIR}/nxc train -i $TRAIN_FILE -o $MODEL $TRAIN_ARGS | tee $TRAIN_RESULT_FILE
    echo "Train date: $(date)" | tee -a $TRAIN_RESULT_FILE
    echo "Model file size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)" | tee -a $TRAIN_RESULT_FILE
    echo "Model file size (K): $(du -c ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+' -o)" | tee -a $TRAIN_RESULT_FILE
    rm -f $TRAIN_LOCK_FILE
fi

# Test model
TEST_ON_TRAIN=0
TEST_USING_NXC_TEST=0

TEST_RESULT_FILE=${RESULTS_DIR}/${TEST_CONFIG}
TEST_LOCK_FILE=${RESULTS_DIR}/.test_lock_${TEST_CONFIG}
if [[ ! -e $TEST_RESULT_FILE ]] || [[ -e $TEST_LOCK_FILE ]]; then
    mkdir -p $RESULTS_DIR
    touch $TEST_LOCK_FILE
    if [ -e $TRAIN_RESULT_FILE ]; then
        cat $TRAIN_RESULT_FILE > $TEST_RESULT_FILE
    fi

    if [[ $TEST_ARGS == *"--labelsWeights"* ]]; then
        TEST_ARGS="${TEST_ARGS} --labelsWeights ${INV_PS_FILE}"
    fi

    PRED_CONFIG=$(echo "${TEST_ARGS}" | tr " /" "__")
    PRED_FILE=${MODEL}/pred_${PRED_CONFIG}
    PRED_LOCK_FILE=${MODEL}/.test_lock_${PRED_CONFIG}
    PRED_RESULT_FILE=${MODEL}/pred_results_${PRED_CONFIG}
    if [[ ! -e $PRED_FILE ]] || [[ -e $PRED_LOCK_FILE ]]; then
        touch $PRED_LOCK_FILE
        ${ROOT_DIR}/nxc test -i $TEST_FILE -o $MODEL $TEST_ARGS --prediction $PRED_FILE --measures "" | tee -a $PRED_RESULT_FILE
        rm -rf $PRED_LOCK_FILE
    fi

    if [ -e $PRED_RESULT_FILE ]; then
        cat $PRED_RESULT_FILE >> $TEST_RESULT_FILE
    fi

    python3 ${SCRIPT_DIR}/evaluate.py $TEST_FILE $PRED_FILE $INV_PS_FILE | tee -a $TEST_RESULT_FILE

    echo "Test date: $(date)" | tee -a $TEST_RESULT_FILE
    rm -rf $TEST_LOCK_FILE
else
    cat $TEST_RESULT_FILE
fi
