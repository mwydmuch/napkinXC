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

BASE_TRAIN_ARGS="-m plt --eps 0.1"
BASE_TEST_ARGS="--topK 5"

for t in "${TREES[@]}"; do
    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} ${t} -c 12 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} ${t} -c 8 --autoCLin 1 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh deliciousLarge "${BASE_TRAIN_ARGS} ${t} -c 1 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC "${BASE_TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${BASE_TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon "${BASE_TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon-3M "${BASE_TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    done
done

BASE_TRAIN_ARGS="-m plt --eps 0.1 --solver L2R_L2LOSS_SVC_DUAL -c 1"
for t in "${TREES[@]}"; do
    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex "${BASE_TRAIN_ARGS} ${t} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10 "${BASE_TRAIN_ARGS} ${t} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat "${BASE_TRAIN_ARGS} ${t} --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh deliciousLarge "${BASE_TRAIN_ARGS} ${t} -c 1 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC "${BASE_TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${BASE_TRAIN_ARGS} ${t} -c 32 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon "${BASE_TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon-3M "${BASE_TRAIN_ARGS} ${t} -c 8 --seed ${s}" "${BASE_TEST_ARGS}" $MODEL $RESULTS
    done
done