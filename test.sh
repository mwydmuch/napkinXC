#!/usr/bin/env bash

DATASET_NAME=$1
MODEL=models/${DATASET_NAME}
DATASET=data/${DATASET_NAME}/${DATASET_NAME}

#rm -r $MODEL
mkdir -p $MODEL

time ./nxml train -i ${DATASET}_train -m $MODEL -t 7 --arity 8
echo

time ./nxml test -i ${DATASET}_test -m $MODEL --topK 5
echo

echo "Model dir: ${MODEL}"
echo "Files (nodes): $(ls ${MODEL} | wc -l | grep -E '[0-9]+' -o)"
echo "Model size: $(du -ch ${MODEL} | tail -n 1 | grep -E '[0-9]+[MB]' -o)"
