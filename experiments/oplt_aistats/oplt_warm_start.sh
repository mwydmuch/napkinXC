#!/usr/bin/env bash

SLURM_JOB=$( dirname "${BASH_SOURCE[0]}" )/../../utils/slurm_job.sh

MODEL=models_oplt_warm_start
RESULTS=models_oplt_warm_start

SEEDS=(2029 2035 2047 2077 2199)
SPLITS=(85 90 95)
EPOCHS=(1 2 3 4)

for sp in "${SPLITS[@]}"; do
    for se in "${SEEDS[@]}"; do
        bash ${EXP_DIR}/split_dataset_and_remap.sh eurlex ${sp} ${se}
        bash ${EXP_DIR}/split_dataset_and_remap.sh amazonCat ${sp} ${se}
        bash ${EXP_DIR}/split_dataset_and_remap.sh wiki10 ${sp} ${se}
        bash ${EXP_DIR}/split_dataset_and_remap.sh wikiLSHTC ${sp} ${se}
        bash ${EXP_DIR}/split_dataset_and_remap.sh amazon ${sp} ${se}
    done
done

for e in "${EPOCHS[@]}"; do
    TRAIN_ARGS="-m oplt --treeType hierarchicalKmeans --maxLeaves 50 --epochs ${e} -t 1"
    RESUME_ARGS="-m oplt --treeType onlineBestScore --epochs ${e} -t 1"
    TEST_ARGS="--topK 5"

    for sp in "${SPLITS[@]}"; do
        for se in "${SEEDS[@]}"; do
            bash ${EXP_DIR}/test_resume.sh eurlex_--split_${sp}_--seed_${se} "${TRAIN_ARGS}" "${RESUME_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
            bash ${EXP_DIR}/test_resume.sh wiki10_--split_${sp}_--seed_${se} "${TRAIN_ARGS}" "${RESUME_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
            bash ${EXP_DIR}/test_resume.sh amazonCat_--split_${sp}_--seed_${se} "${TRAIN_ARGS}"  "${RESUME_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
            bash ${EXP_DIR}/test_resume.sh wikiLSHTC_--split_${sp}_--seed_${se} "${TRAIN_ARGS}" "${RESUME_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
            bash ${EXP_DIR}/test_resume.sh amazon_--split_${sp}_--seed_${se} "${TRAIN_ARGS}" "${RESUME_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
        done
    done
done
