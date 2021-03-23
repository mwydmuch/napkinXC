#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_hsm_pick_one_label
RESULTS=results_hsm_pick_one_label

SEEDS=(1993 2020 2029 2047 2077)

TRAIN_ARGS="-m hsm --eps 0.1 --pickOneLabelWeighting 1"
TEST_ARGS="--topK 5"
for s in "${SEEDS[@]}"; do
   bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} -C 12 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} -C 16 --autoCLin 1 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} -C 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} -C 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} -C 16 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} -C 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS

   # These require more than 256 GB of RAM to run
   #bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} -C 1 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   #bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} -C 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
done