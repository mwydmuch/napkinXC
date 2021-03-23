#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_oplt
RESULTS=results_oplt

SEEDS=(1993 2020 2029 2047 2077)
EPOCHS=(1 2 3 4)

for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/shuffle_dataset.sh eurlex ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh amazonCat ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh wiki10 ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh wikiLSHTC ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh amazon ${s}
done

for e in "${EPOCHS[@]}"; do
    TRAIN_ARGS="-m oplt --optimizer adagrad --loss log --treeType onlineBestScore --epochs ${e} --onlineTreeAlpha 0.75 -t 1"
    TEST_ARGS="--topK 5"

    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    done

    TRAIN_ARGS="-m oplt --optimizer adagrad --loss log --treeType onlineRandom --epochs ${e} -t 1"

    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    done
done