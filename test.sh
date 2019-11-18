#!/usr/bin/env bash

set -e

DATASET_NAME=$1
shift
ARGS="$@"
MODEL=models/${DATASET_NAME}_$(echo "${ARGS}" | tr " " "_")
TRAIN_LOCK_FILE=$MODEL/.train_lock
DATASET_DIR=data/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}

if [ ! -e $DATASET_DIR ]; then
    bash get_data.sh $DATASET_NAME
fi

if [ -e "${DATASET_FILE}.train.remapped" ]; then
    TRAIN_FILE="${DATASET_FILE}.train.remapped"
    TEST_FILE="${DATASET_FILE}.test.remapped"
elif [ -e "${DATASET_FILE}_train.txt" ]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt"
    TEST_FILE="${DATASET_FILE}_test.txt"
elif [ -e "${DATASET_FILE}.train" ]; then
    TRAIN_FILE="${DATASET_FILE}.train"
    TEST_FILE="${DATASET_FILE}.test"
fi

if [ ! -e nxc ]; then
    rm -f CMakeCache.txt
    cmake -DCMAKE_BUILD_TYPE=Release
    make -j
fi

if [ -e $TRAIN_LOCK_FILE ]; then
    rm -rf $MODEL
    rm -rf $TRAIN_LOCK_FILE
fi

if [ ! -e $MODEL ]; then
    mkdir -p $MODEL
    touch $TRAIN_LOCK_FILE
    time ./nxc train -i $TRAIN_FILE -o $MODEL -t -1 $ARGS
    echo
fi

rm -rf $TRAIN_LOCK_FILE
time ./nxc test -i $TEST_FILE -o $MODEL --topK 5 -t -1 --ensemble 3
echo

echo "Model dir: ${MODEL}"
echo "Model size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)"
