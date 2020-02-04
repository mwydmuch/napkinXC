#!/usr/bin/env bash

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )

createOneShotLearningDataset () {
    local DATASET_NAME=$1
    local DATASET_DIR=data/${DATASET_NAME}
    local ONESHOT_DATASET_DIR=${DATASET_DIR}1Shot
    local DATASET_FILE_PREFIX=data/${DATASET_NAME}/${DATASET_NAME}
    local ONESHOT_DATASET_FILE_PREFIX=${ONESHOT_DATASET_DIR}/${DATASET_NAME}1Shot

    echo "Creating 1-shot learning dataset for ${DATASET_NAME}"
    mkdir -p ${ONESHOT_DATASET_DIR}
    gawk '{if (NR > 1){$1=""; print (NR-2) $0}}' ${DATASET_FILE_PREFIX}_train.txt > ${ONESHOT_DATASET_FILE_PREFIX}.train
    cp ${ONESHOT_DATASET_FILE_PREFIX}.train ${ONESHOT_DATASET_FILE_PREFIX}.test
}

createKNNUnsupDataset () {
    local DATASET_NAME=$1
    local DATASET_DIR=data/${DATASET_NAME}
    local KNN_DATASET_DIR=${DATASET_DIR}KNNUnsup
    local DATASET_FILE_PREFIX=data/${DATASET_NAME}/${DATASET_NAME}
    local KNN_DATASET_FILE_PREFIX=${KNN_DATASET_DIR}/${DATASET_NAME}KNNUnsup

    echo "Creating k-NN learning dataset for ${DATASET_NAME}"
    mkdir -p ${KNN_DATASET_DIR}
    gawk '{if (NR > 1){$1=""; print (NR-2) $0}}' ${DATASET_FILE_PREFIX}_train.txt > ${KNN_DATASET_FILE_PREFIX}.train
    tail -n +2  ${DATASET_FILE_PREFIX}_test.txt > ${KNN_DATASET_FILE_PREFIX}.test.tmp
    python ${SCRIPT_DIR}/calculate_knn.py ${KNN_DATASET_FILE_PREFIX}.train ${KNN_DATASET_FILE_PREFIX}.test.tmp ${KNN_DATASET_FILE_PREFIX}.test
    rm ${KNN_DATASET_FILE_PREFIX}.test.tmp
}

createKNNSupDataset () {
    local DATASET_NAME=$1
    local DATASET_DIR=data/${DATASET_NAME}
    local KNN_DATASET_DIR=${DATASET_DIR}KNNSup
    local DATASET_FILE_PREFIX=data/${DATASET_NAME}/${DATASET_NAME}
    local KNN_DATASET_FILE_PREFIX=${KNN_DATASET_DIR}/${DATASET_NAME}KNNSup

    echo "Creating k-NN learning dataset for ${DATASET_NAME}"
    mkdir -p ${KNN_DATASET_DIR}
    gawk '{if (NR > 1){$1=""; print (NR-2) $0}}' ${DATASET_FILE_PREFIX}_train.txt > ${KNN_DATASET_FILE_PREFIX}.train.tmp
    tail -n +2  ${DATASET_FILE_PREFIX}_test.txt > ${KNN_DATASET_FILE_PREFIX}.test.tmp
    python ${SCRIPT_DIR}/calculate_knn.py ${KNN_DATASET_FILE_PREFIX}.train.tmp ${KNN_DATASET_FILE_PREFIX}.train.tmp ${KNN_DATASET_FILE_PREFIX}.train
    python ${SCRIPT_DIR}/calculate_knn.py ${KNN_DATASET_FILE_PREFIX}.train.tmp ${KNN_DATASET_FILE_PREFIX}.test.tmp ${KNN_DATASET_FILE_PREFIX}.test
    rm ${KNN_DATASET_FILE_PREFIX}.train.tmp
    rm ${KNN_DATASET_FILE_PREFIX}.test.tmp
}

#createOneShotLearningDataset eurlex
#createKNNUnsupDataset eurlex
#createKNNSupDataset eurlex
#
#createOneShotLearningDataset wiki10
#createKNNUnsupDataset wiki10
#createKNNSupDataset wiki10

mv ${SCRIPT_DIR}/../data/Dmoz/Dmoz.train ${SCRIPT_DIR}/../data/Dmoz/Dmoz_train.txt
mv ${SCRIPT_DIR}/../data/Dmoz/Dmoz.test ${SCRIPT_DIR}/../data/Dmoz/Dmoz_test.txt

createOneShotLearningDataset Dmoz
createKNNUnsupDataset Dmoz
createKNNSupDataset Dmoz

mv ${SCRIPT_DIR}/../data/imageNet/imageNet.train ${SCRIPT_DIR}/../data/imageNet/imageNet_train.txt
mv ${SCRIPT_DIR}/../data/imageNet/imageNet.test ${SCRIPT_DIR}/../data/imageNet/imageNet_test.txt

createOneShotLearningDataset imageNet
createKNNUnsupDataset imageNet
createKNNSupDataset imageNet
