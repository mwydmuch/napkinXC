#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_plt_optimzation
RESULTS=results_plt_optimzation

SEEDS=(1993 2020 2029 2047 2077)

test_all () {
    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    done
}

TRAIN_ARGS="-m plt --eps 0.1 --treeType hierarchicalKmeans"
TEST_ARGS="--topK 5"
for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} -C 12 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} -C 16 --autoCLin 1 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} -C 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} -C 1 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} -C 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} -C 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} -C 16 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} -C 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
done

TRAIN_ARGS="-m plt --eps 0.1 --solver L2R_L2LOSS_SVC_DUAL -C 1 --treeType hierarchicalKmeans"
test_all

TRAIN_ARGS="-m plt --optimizer adagrad --loss log --adagradEps 0.01 --epochs 3 --treeType hierarchicalKmeans"
test_all

TRAIN_ARGS="-m plt --optimizer adagrad --loss log --adagradEps 0.001 --epochs 3 --treeType hierarchicalKmeans"
test_all

TRAIN_ARGS="-m plt --optimizer adagrad --loss l2 --adagradEps 0.01 --epochs 3 --treeType hierarchicalKmeans"
test_all

TRAIN_ARGS="-m plt --optimizer adagrad --loss l2 --adagradEps 0.001 --epochs 3 --treeType hierarchicalKmeans"
test_all
