#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_plt_final
RESULTS=results_plt_final

SEEDS=(1993 2020 2029 2047 2077)

BASE_TRAIN_ARGS="-m plt --eps 0.1 --ensemble 3 --arity 16 --kMeansBalanced 0"
BASE_TEST_ARGS="--topK 5 --ensemble 3 --onTheTrotPrediction 1"
for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} -C 12 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} -C 16 --autoCLin 1 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} -C 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh deliciousLarge "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wikiLSHTC "${BASE_TRAIN_ARGS} -C 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon "${BASE_TRAIN_ARGS} -C 16 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon-3M "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
done

BASE_TRAIN_ARGS="-m plt --eps 0.1 --ensemble 3 --solver L2R_L2LOSS_SVC_DUAL -c 1 --arity 16 --kMeansBalanced 0"
for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh deliciousLarge "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wikiLSHTC "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon-3M "${BASE_TRAIN_ARGS} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
done
