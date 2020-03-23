#!/usr/bin/env bash

set -e
set -o pipefail

DATASET_NAME=$1
MODEL_DIR=models
RESULTS_DIR=results

# If there are exactly 3 arguments that starts with nxc parameter (-)
if [[ $# -gt 2 ]] && [[ $2 == -* ]] && [[ $3 == -* ]]; then
    TRAIN_ARGS=$2
    OFO_ARGS=$3
    if [[ $# -gt 3 ]]; then
        MODEL_DIR=$4
    fi
    if [[ $# -gt 4 ]]; then
        RESULTS_DIR=$5
    fi
else
    shift
    TRAIN_ARGS="$@"
    OFO_ARGS=""
fi

TRAIN_CONFIG=${DATASET_NAME}_$(echo "${TRAIN_ARGS}" | tr " /" "__")
OFO_CONFIG=${TRAIN_CONFIG}_$(echo "${OFO_ARGS}" | tr " /" "__")

MODEL=${MODEL_DIR}/${TRAIN_CONFIG}
DATASET_DIR=data/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}

# Download dataset
if [[ ! -e $DATASET_DIR ]]; then
    bash scripts/get_data.sh $DATASET_NAME
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
fi

# Build nxc
if [[ ! -e nxc ]]; then
    rm -f CMakeCache.txt
    cmake -DCMAKE_BUILD_TYPE=Release .
    make -j
fi

# Train model
TRAIN_RESULT_FILE=${MODEL}/train_results
TRAIN_LOCK_FILE=${MODEL}/.train_lock
if [[ ! -e $MODEL ]] || [[ -e $TRAIN_LOCK_FILE ]]; then
    mkdir -p $MODEL
    touch $TRAIN_LOCK_FILE
    (time ./nxc train $TRAIN_ARGS -i $TRAIN_FILE -o $MODEL | tee $TRAIN_RESULT_FILE)
    echo
    echo "Train date: $(date)" | tee -a $TRAIN_RESULT_FILE
    rm -rf $TRAIN_LOCK_FILE
fi

# Optimized thresholds
THRESHOLDS_FILE=${MODEL}/ofo_tresholds_$(echo "${OFO_ARGS}" | tr " /" "__")
OFO_LOCK_FILE=${MODEL}/.ofo_lock_${OFO_CONFIG}
if [[ ! -e $THRESHOLDS_FILE ]] || [[ -e $OFO_LOCK_FILE ]]; then
    touch $OFO_LOCK_FILE
    (time ./nxc ofo $OFO_ARGS -i $TEST_FILE -o $MODEL --thresholds $THRESHOLDS_FILE)
    rm -rf $OFO_LOCK_FILE
fi

## Test model
TEST_RESULT_FILE=${RESULTS_DIR}/ofo_${OFO_CONFIG}
TEST_LOCK_FILE=${RESULTS_DIR}/.ofo_test_lock_${OFO_CONFIG}
if [[ ! -e $TEST_RESULT_FILE ]] || [[ -e $TEST_LOCK_FILE ]]; then
    mkdir -p $RESULTS_DIR
    touch $TEST_LOCK_FILE
    if [ -e $TRAIN_RESULT_FILE ]; then
        cat $TRAIN_RESULT_FILE > $TEST_RESULT_FILE
    fi

    (time ./nxc test -i $TEST_FILE -o $MODEL --thresholds $THRESHOLDS_FILE --measures hl,microf,macrof | tee -a $TEST_RESULT_FILE)

    echo
    echo "Model file size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)" | tee -a $TEST_RESULT_FILE
    echo "Test date: $(date)" | tee -a $TEST_RESULT_FILE
    rm -rf $TEST_LOCK_FILE
else
    cat $TEST_RESULT_FILE
fi
