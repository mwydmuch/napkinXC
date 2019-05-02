#!/usr/bin/env bash

#bash remap_datasets.sh

DATASETS=(sector aloi.bin imageNet LSHTC1 Dmoz)

for d in "${DATASETS[@]}"; do
    echo "${d} tests..."
    bash testBOP.sh $d -m hsmubop --treeType completeRandom --header 0
    bash testBOP.sh $d -m hsmubop --treeType hierarchicalKMeans --header 0 --maxLeaves 16
    bash testBOP.sh $d -m ubop --header 0
done
