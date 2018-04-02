#!/usr/bin/env bash

DATASET="$1"

case "$DATASET" in
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
    *)
        echo "Unknown dataset: ${DATASET}!"
        exit 1
esac

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )

mkdir -p data
DATASET_PATH="${SCRIPT_DIR}/data/${DATASET}"

if [ ! -e $DATASET_PATH ]; then
    echo "Getting $DATASET ..."
    if [ ! -e ${DATASET_PATH}.zip ]; then
        echo "Downloading ${DATASET_PATH}.zip ..."
        perl ${SCRIPT_DIR}/google_drive_download.pl $DATASET_LINK ${DATASET_PATH}.zip
    fi

    echo "Extracting ${DATASET_PATH} ..."
    unzip -j -d $DATASET_PATH ${DATASET_PATH}.zip
    rm ${DATASET_PATH}.zip
fi
