#!/usr/bin/env bash

set -e

DATASET="$1"

case "$DATASET" in
    # Multi-label datasets from The Extreme Classification Repository (manikvarma.org/downloads/XC/XMLRepository.html)
    "amazonCat") # AmazonCat-13K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGa2tMbVJGdDNSMGc" ;;
    "amazonCat-14K") # AmazonCat-14K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDFqU2E5U0dxS00" ;;
    "amazon") # Amazon-670K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGdUJwRzltS1dvUVk" ;;
    "amazon-3M") # Amazon-3M
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGUEd4eTRxaWl3YkE" ;;
    "deliciousLarge") # Delicious-200K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGR3lBWWYyVlhDLWM" ;;
    "eurlex") # EURLex-4K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGU0VTR1pCejFpWjg" ;;
    "wiki10") # Wiki10-31K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGaDdOeGliWF9EOTA" ;;
    "wikiLSHTC") # WikiLSHTC-325K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGSHE1SWx4TVRva3c" ;;
    "WikipediaLarge-500K") # Wikipedia-500K
        DATASET_LINK="https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGRmEzVDVkNjBMR3c" ;;

    # Multi-class datasets from PD-Sparse (http://www.cs.utexas.edu/~xrhuang/PDSparse/)
    "bibtex")
	    DATASET_LINK="https://www.dropbox.com/s/yizjmqxi6ulub3y/bibtex.tar.bz2" ;;
	"sector")
	    DATASET_LINK="https://www.dropbox.com/s/9qhaxn9rxb0n4mq/sector.tar.bz2" ;;
	"rcv1_regions")
        DATASET_LINK="https://www.dropbox.com/s/be6sit7zcxxy4uw/rcv1_regions.tar.bz2" ;;
	"aloi.bin")
	    DATASET_LINK="https://www.dropbox.com/s/jwousxredtggq3y/aloi.bin.tar.bz2" ;;
	"Dmoz")
	    DATASET_LINK="https://www.dropbox.com/s/lq6vd7t6nz1w7iy/Dmoz.tar.bz2" ;;
    "Eur-Lex")
	    DATASET_LINK="https://www.dropbox.com/s/swjyqbtxea1q1mp/Eur-Lex.tar.bz2" ;;
    "LSHTC1")
	    DATASET_LINK="https://www.dropbox.com/s/0mgarujfps4cb78/LSHTC1.tar.bz2" ;;
    "LSHTCwiki")
	    DATASET_LINK="https://www.dropbox.com/s/3j8urvy6j9kyx4u/LSHTCwiki.tar.bz2" ;;
    "imageNet")
        DATASET_LINK="https://www.dropbox.com/s/898cgz76lmuz04a/imageNet.tar.bz2" ;;
    *)
        echo "Unknown dataset: ${DATASET}!"
        exit 1
esac

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )

mkdir -p data
DATA_DIR=${SCRIPT_DIR}/data
DATASET_PATH="${DATA_DIR}/${DATASET}"

if [ ! -e $DATASET_PATH ]; then
    echo "Getting $DATASET ..."

    if [ ! -e ${DATASET_PATH}.* ]; then
        if [[ $DATASET_LINK == *drive.google.com* ]]; then
            echo "Downloading ${DATASET_PATH}.zip ..."
            perl ${SCRIPT_DIR}/google_drive_download.pl $DATASET_LINK ${DATASET_PATH}.zip
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
