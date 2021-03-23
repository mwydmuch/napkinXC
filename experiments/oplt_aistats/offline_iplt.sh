#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_iplt
RESULTS=results_iplt

SEEDS=(1993 2020 2029 2047 2077)
EPOCHS=(1 2 3 4)

for e in "${EPOCHS[@]}"; do
    TRAIN_ARGS="-m plt --optimizer adagrad --treeType hierarchicalKmeans --epochs ${e}"
    TEST_ARGS="--topK 5"

    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    done
done
