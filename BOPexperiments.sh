#!/usr/bin/env bash

#bash remap_datasets.sh

DATASETS=(aloi.bin)

for d in "${DATASETS[@]}"; do
    echo "${d} tests..."
    if [ -e "data/${d}/${d}.hier" ]; then
        bash testBOP.sh $d -m hsmubop --treeStructure "data/${d}/${d}.hier"
    fi
    bash testBOP.sh $d -m hsmubop --treeType completeRandom
    bash testBOP.sh $d -m hsmubop --treeType hierarchicalKMeans --maxLeaves 16
    bash testBOP.sh $d -m ubop
    #bash testBOP.sh $d -m rbop
    #bash testBOP.sh $d -m hsmrbop
done
