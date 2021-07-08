#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_psplt
RESULTS=results_psplt

SEEDS=(1993)

# log loss (liblinear optimizer)
BASE_TRAIN_ARGS="-m br --loss log"
BASE_TEST_ARGS="--topK 5"
for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
done

# squared hinge loss (a bit worse than log loss in terms of PSP@k, but with faster to train)
BASE_TRAIN_ARGS="-m br --loss squaredHinge -C 1"
for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
done
