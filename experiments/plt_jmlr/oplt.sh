#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_oplt
RESULTS=results_oplt

SEEDS=(1993 2020 2029 2047 2077)

TRAIN_ARGS="-m oplt --optimizer adagrad --epochs 3 --treeType onlineKaryComplete --arity 2 -t 1"
TEST_ARGS="--topK 5"

for s in "${SEEDS[@]}"; do
   bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS

   bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} --hash 13000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} --hash 26000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} --hash 52000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS

   bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} --hash 8000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} --hash 16000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} --hash 32000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS

   bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} --hash 5000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} --hash 10000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} --hash 20000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS

   bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} --hash 4000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} --hash 8000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} --hash 16000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS

   bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} --hash 1000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} --hash 2000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
   bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} --hash 4000 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS

   # This one may need more than 256 GB of RAM
   #bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
done
