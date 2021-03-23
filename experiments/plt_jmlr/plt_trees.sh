#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_plt_trees
RESULTS=results_plt_tress

SEEDS=(1993 2020 2029 2047 2077)
TREES=(
    "--treeType huffman --arity 2"
    "--treeType completeRandom --arity 2"
    "--treeType balancedRandom --arity 16"
    "--treeType balancedRandom --arity 64"
    "--treeType balancedRandom --maxLeaves 25"
    "--treeType balancedRandom --maxLeaves 100"
    "--treeType balancedRandom --maxLeaves 400"
    "--treeType hierarchicalKmeans --arity 2"
    "--treeType hierarchicalKmeans --arity 16"
    "--treeType hierarchicalKmeans --arity 64"
    "--treeType hierarchicalKmeans --arity 2 --kmeansBalanced 0"
    "--treeType hierarchicalKmeans --arity 16 --kmeansBalanced 0"
    "--treeType hierarchicalKmeans --arity 64 --kmeansBalanced 0"
    "--treeType hierarchicalKmeans --maxLeaves 25"
    "--treeType hierarchicalKmeans --maxLeaves 100"
    "--treeType hierarchicalKmeans --maxLeaves 400"
)
TRAIN_ARGS="-m plt --eps 0.1"
TEST_ARGS="--topK 5"

for t in "${TREES[@]}"; do
    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} ${t} -c 12 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} ${t} -c 8 --autoCLin 1 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} ${t} -c 1 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    done
done

TRAIN_ARGS="-m plt --eps 0.1 --solver L2R_L2LOSS_SVC_DUAL -C 1"
for t in "${TREES[@]}"; do
    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} ${t} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} ${t} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} ${t} --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} ${t} -c 1 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${TEST_ARGS}" $MODEL $RESULTS
    done
done