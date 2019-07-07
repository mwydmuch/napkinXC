#!/usr/bin/env bash

set -e

DATASET_NAME=$1
shift
ARGS="$@"
MODEL_NAME="${DATASET_NAME}_$(echo "${ARGS}" | tr " /" "__")"
MODEL=models/${MODEL_NAME}
DATASET_DIR=data/${DATASET_NAME}
DATASET_FILE=${DATASET_DIR}/${DATASET_NAME}
THREADS=-1

OUTPUT=outputs/${MODEL_NAME}
mkdir -p outputs

RESULTS=results/${MODEL_NAME}
mkdir -p results


U=("uP" "uAlfaBeta --alfa 1.0 --beta 0.1" "uAlfaBeta --alfa 1.0 --beta 0.5" "uAlfaBeta --alfa 1.0 --beta 1.0")


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

if [ ! -e nxml ]; then
    rm -f CMakeCache.txt
    cmake -DCMAKE_BUILD_TYPE=Release
    make -j
fi

#rm -rf $MODEL
if [ ! -e $MODEL ]; then
    echo "Train ..."
    mkdir -p $MODEL
    (time ../nxml train -i $TRAIN_FILE -o $MODEL -t $THREADS $ARGS --header 0) > ${OUTPUT}_train 2>&1
    echo "Model dir: ${MODEL}" >> ${OUTPUT}_train 2>&1
    echo "Model size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)" >> ${OUTPUT}_train 2>&1
fi

echo "${MODEL_NAME}" > ${RESULTS}
echo "Train ${u}" >> ${RESULTS}
grep "Mean # estimators per data point" ${OUTPUT}_train >> ${RESULTS}
grep "Model size" ${OUTPUT}_train >> ${RESULTS}
grep "user" ${OUTPUT}_train >> ${RESULTS}
grep "real" ${OUTPUT}_train >> ${RESULTS}
echo >> ${RESULTS}

for u in "${U[@]}"; do
    base_u=$(echo ${u} | cut -f 1 -d " ")
    full_u=$(echo "${u}" | tr " " "_")
    echo "Test with utility ${u} ..."

    if [ ! -e ${OUTPUT}_test_${full_u} ]; then
        (time ../nxml test -i $TEST_FILE -o $MODEL -t $THREADS --header 0 --setBasedU ${u}) > ${OUTPUT}_test_${full_u} 2>&1
    fi

    echo "Test ${u}" >> ${RESULTS}
    grep "Recall" ${OUTPUT}_test_${full_u} >> ${RESULTS}
    grep ${base_u} ${OUTPUT}_test_${full_u} >> ${RESULTS}
    grep "Mean pred. size" ${OUTPUT}_test_${full_u} >> ${RESULTS}
    grep "Mean # estimators per data point" ${OUTPUT}_test_${full_u} >> ${RESULTS}
    grep "user" ${OUTPUT}_test_${full_u} >> ${RESULTS}
    grep "real" ${OUTPUT}_test_${full_u} >> ${RESULTS}
    echo >> ${RESULTS}
done

