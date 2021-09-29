#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_psplt
RESULTS=results_psplt

SEEDS=(1993 2021 2029 2047 2077)

# log loss (liblinear optimizer), ensemble of 3 trees
BASE_TRAIN_ARGS="-m plt --ensemble 3 --loss log"
# difference between propensity scored and vanila version is in "--labelsWeights 1" argument
BASE_TEST_ARGS="--topK 5 --ensemble 3 --labelsWeights 1"
for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} -C 16 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} -C 16 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} -C 16 --autoCLin 1 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} -C 16 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wikiLSHTC "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
done

# squared hinge loss (a bit worse than log loss in terms of PSP@k, but with faster to train)
BASE_TRAIN_ARGS="-m plt --ensemble 3 --loss squaredHinge -C 1"
for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wikiLSHTC "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
done
