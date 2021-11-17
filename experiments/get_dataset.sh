#!/usr/bin/env bash

# This script download only datasets in BOW format.

set -e

DATASET="$1"
DATA_DIR="data"
if [[ $# -gt 2 ]]; then
    DATA_DIR="$2"
fi

case "$DATASET" in
    # Small datasets used for debug purposed
    "yeast") # yeast
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1qcU-jxF7VmHBvwTn0-6MB4FW4ItD-bIp" ;;
    "mediamill") # mediamill
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1YDlU9GTYSoydU1UYTNT8BpPkRw1BCnFw" ;;

    # Multi-label datasets from The Extreme Classification Repository (manikvarma.org/downloads/XC/XMLRepository.html)
    "rcv1x") # RCV1X-2K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1WvBWF9hH5ZlFcpCRhp8wLVGwdeGZXKD2" ;;
    "amazonCat") # AmazonCat-13K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=17EFQtnRswEv0XyPng2EOy5IeWmeZbe0a" ;;
    "amazonCat-14K") # AmazonCat-14K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=19IhTX1_a4U5I2Q56VUkzq2jFBpwm2-n-" ;;
    "amazon") # Amazon-670K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1PV-wbKVv6Ng1K1XM1USqNxFhaQo2qfk6" ;;
    "amazon-3M") # Amazon-3M
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1ork7yeAcliD9JQiRdEx4QknhMFHjVrku" ;;
    "deliciousLarge") # Delicious-200K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=18Tb595TuGyFW--nEYGfduwS5WvSbX-sX" ;;
    "eurlex") # EURLex-4K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1xWW1UykQBTD2IorVn7V8XubJatQDScol" ;;
    "wiki10") # Wiki10-31K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1uV-p682ggXQQTiKZyK5M6B0xJJk8_lJx" ;;
    "wikiLSHTC") # WikiLSHTC-325K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1Y5xEJu-j3M7voyFV5Ouq9Wl3vyd9fus2" ;;
    "WikipediaLarge-500K") # Wikipedia-500K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=1LkGdINF5coOfkm6J2xys3V1NMW7zz8U1" ;;
    *)
        echo "Unknown dataset: ${DATASET}!"
        exit 1
esac

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
mkdir -p $DATA_DIR
DATASET_PATH="${DATA_DIR}/${DATASET}"

if [ ! -e $DATASET_PATH ]; then
    echo "Getting $DATASET ..."

    if [ ! -e ${DATASET_PATH}.* ]; then
        if [[ $DATASET_LINK == *drive.google.com* ]]; then
            echo "Downloading ${DATASET_PATH}.zip ..."
            GOOGLE_ID=$(echo ${DATASET_LINK} | grep -o '[^=]*$')
            curl -c ${DATASET_PATH}.cookie -s -L "https://drive.google.com/uc?export=download&id=${GOOGLE_ID}" > /dev/null
            curl -Lb ${DATASET_PATH}.cookie "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ${DATASET_PATH}.cookie)&id=${GOOGLE_ID}" -o ${DATASET_PATH}.zip
            rm ${DATASET_PATH}.cookie
        elif [[ $DATASET_LINK == *dropbox.com* ]]; then
            echo "Downloading $(basename ${DATASET_LINK}) ..."
            wget $DATASET_LINK -P $DATA_DIR
        fi
    fi

    echo "Extracting ${DATASET_PATH} ..."
    if [[ -e ${DATASET_PATH}.zip ]]; then
        unzip -j -d $DATASET_PATH ${DATASET_PATH}.zip
        rm ${DATASET_PATH}.zip
    elif [[ -e ${DATASET_PATH}.tar.bz2 ]]; then
        mkdir -p $DATA_DIR
        tar xjC $DATA_DIR -f ${DATASET_PATH}.tar.bz2
        rm ${DATASET_PATH}.tar.bz2
    fi
fi

