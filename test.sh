#!/usr/bin/env bash

DATASET_NAME=$1
shift
ARGS="$@"
MODEL=models/${DATASET_NAME}_$(echo "${ARGS}" | tr " " "_")
DATASET=data/${DATASET_NAME}/${DATASET_NAME}

if [ ! -e "data/${DATASET_NAME}" ]; then
    bash get_data.sh $DATASET_NAME
fi

if [ ! -e nxml ]; then
    rm CMakeCache.txt
    cmake -DCMAKE_BUILD_TYPE=Release
    make
fi

if [ ! -e $MODEL ]; then
    mkdir -p $MODEL
    time ./nxml train -i ${DATASET}_train.txt -m $MODEL -t -1 $ARGS
    echo
fi

time ./nxml test -i ${DATASET}_test.txt -m $MODEL --topK 5 -t -1
echo

echo "Model dir: ${MODEL}"
echo "Model size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9\.,]+[BMG]' -o)"
