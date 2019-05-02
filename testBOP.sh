#!/usr/bin/env bash

set -e

DATASET_NAME=$1
shift
ARGS="$@"
MODEL=models/${DATASET_NAME}_$(echo "${ARGS}" | tr " " "_")
DATASET_DIR=data/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}

RESULTS=results/${DATASET_NAME}_$(echo "${ARGS}" | tr " " "_")
mkdir -p results

U=(uP uF1 uAlfaBeta)

if [ ! -e $DATASET_DIR ]; then
    bash get_data.sh $DATASET_NAME
fi

if [ -e "${DATASET_FILE}_train.txt" ]; then
    TRAIN_FILE="${DATASET_FILE}_train.txt"
    TEST_FILE="${DATASET_FILE}_test.txt"
elif [ -e "${DATASET_FILE}.train" ]; then
    TRAIN_FILE="${DATASET_FILE}.train"
    TEST_FILE="${DATASET_FILE}.test"
fi

if [ ! -e nxml ]; then
    rm -f CMakeCache.txt
    cmake -DCMAKE_BUILD_TYPE=Release
    make -j
fi

rm -rf $MODEL
if [ ! -e $MODEL ]; then
    echo "Train ..."
    mkdir -p $MODEL
    (time ./nxml train -i $TRAIN_FILE -o $MODEL -t -1 $ARGS) > ${RESULTS} 2>&1
fi


for u in "${U[@]}"; do
    echo "Utility ${u}..."
    (time ./nxml test -i $TEST_FILE -o $MODEL -t -1 --setBasedU ${u}) >> ${RESULTS} 2>&1
done


echo "Model dir: ${MODEL}" >> ${RESULTS} 2>&1
echo "Model size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)" >> ${RESULTS} 2>&1
