#!/usr/bin/env bash

EXP_DIR=$( dirname "${BASH_SOURCE[0]}" )/..

MODEL=models_plt_ofo
RESULTS=results_plt_ofo

SEEDS=(1993 2020 2029 2047 2077)

for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/split_dataset.sh eurlex 30 ${s}
    bash ${EXP_DIR}/split_dataset.sh amazonCat 30 ${s}
    bash ${EXP_DIR}/split_dataset.sh wiki10 30 ${s}
    bash ${EXP_DIR}/split_dataset.sh deliciousLarge 30 ${s}
    bash ${EXP_DIR}/split_dataset.sh wikiLSHTC 30 ${s}
    bash ${EXP_DIR}/split_dataset.sh WikipediaLarge-500K 30 ${s}
    bash ${EXP_DIR}/split_dataset.sh amazon 30 ${s}
    bash ${EXP_DIR}/split_dataset.sh amazon-3M 30 ${s}
done

TRAIN_ARGS="-m plt --eps 0.1"
TEST_ARGS="--measures p@1,hl,microf1,macrof1,samplef1,s"
OFO_ARGS="--ofoType micro --ofoA 10 --ofoB 20 --epochs 1"

for s in "${SEEDS[@]}"; do
    bash ${EXP_DIR}/test_ofo.sh eurlex_--split_30_--seed_${s} "${TRAIN_ARGS} -c 12" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test_ofo.sh wiki10_--split_30_--seed_${s} "${TRAIN_ARGS} -c 16" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test_ofo.sh amazonCat_--split_30_--seed_${s} "${TRAIN_ARGS} -c 8" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test_ofo.sh deliciousLarge_--split_30_--seed_${s} "${TRAIN_ARGS} -c 1" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test_ofo.sh wikiLSHTC_--split_30_--seed_${s} "${TRAIN_ARGS} -c 32" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test_ofo.sh WikipediaLarge-500K_--split_30_--seed_${s} "${TRAIN_ARGS} -c 32" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test_ofo.sh amazon_--split_30_--seed_${s} "${TRAIN_ARGS} -c 16" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test_ofo.sh amazon-3M_--split_30_--seed_${s} "${TRAIN_ARGS} -c 8" " ${OFO_ARGS}" "${TEST_ARGS}" $MODEL $RESULTS

    # k = avg. #labels per example
    bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} -c 12 --seed ${s}" "--topK 5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} -c 16 --seed ${s}" "--topK 5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} -c 8 --seed ${s}" "--topK 19 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} -c 1 --seed ${s}" "--topK 76 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} -c 32 --seed ${s}" "--topK 3 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} -c 32 --seed ${s}" "--topK 5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} -c 16 --seed ${s}" "--topK 5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} -c 8 --seed ${s}" "--topK 36 ${TEST_ARGS}" $MODEL $RESULTS

    # prediction with threshold = 0.5
    bash ${EXP_DIR}/test.sh eurlex "${TRAIN_ARGS} -c 12 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wiki10 "${TRAIN_ARGS} -c 16 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazonCat "${TRAIN_ARGS} -c 8 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh deliciousLarge "${TRAIN_ARGS} -c 1 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh wikiLSHTC "${TRAIN_ARGS} -c 32 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh WikipediaLarge-500K "${TRAIN_ARGS} -c 32 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon "${TRAIN_ARGS} -c 16 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
    bash ${EXP_DIR}/test.sh amazon-3M "${TRAIN_ARGS} -c 8 --seed ${s}" "--threshold 0.5 ${TEST_ARGS}" $MODEL $RESULTS
done
