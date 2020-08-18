#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_hsm_pick_one_label
RESULTS=results_hsm_pick_one_label

SEEDS=(1993 2020 2029 2047 2077)

BASE_TRAIN_ARGS="-m hsm --eps 0.1 --pickOneLabelWeighting 1"
BASE_TEST_ARGS="--topK 5"
for s in "${SEEDS[@]}"; do
   bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} -C 12 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} -C 16 --autoCLin 1 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} -C 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wikiLSHTC "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazon "${BASE_TRAIN_ARGS} -C 16 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS

   # These require more than 256 GB of RAM to run
   #bash ${EXP_DIR}/test.sh deliciousLarge "${BASE_TRAIN_ARGS} -C 1 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
   #bash ${EXP_DIR}/test.sh amazon-3M "${BASE_TRAIN_ARGS} -C 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
done