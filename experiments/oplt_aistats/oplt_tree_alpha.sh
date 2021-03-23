#!/usr/bin/env bash

SLURM_JOB=$( dirname "${BASH_SOURCE[0]}" )/../../utils/slurm_job.sh

MODEL=models_oplt
RESULTS=results_oplt_tree_alpha

SEEDS=(1993 2020 2029 2047 2077)
ALPHA=("0.125" "0.25" "0.375" "0.5" "0.625" "0.75" "0.875")

for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/shuffle_dataset.sh eurlex ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh amazonCat ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh wiki10 ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh wikiLSHTC ${s}
    bash ${EXP_DIR}/shuffle_dataset.sh amazon ${s}
done

for a in "${ALPHA[@]}"; do
    TRAIN_ARGS="-m oplt --optimizer adagrad --loss log --treeType onlineBestScore --epochs 3 --onlineTreeAlpha ${a} -t 1"
    TEST_ARGS="--topK 5"

    for s in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/test.sh eurlex_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wiki10_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazonCat_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh wikiLSHTC_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        bash ${EXP_DIR}/test.sh amazon_--seed_${s} "${TRAIN_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    done
done
