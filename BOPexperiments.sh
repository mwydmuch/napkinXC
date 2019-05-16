#!/usr/bin/env bash

#bash remap_datasets.sh

DATASETS=(LSHTC1 Dmoz)

for d in "${DATASETS[@]}"; do
    echo "${d} tests..."

    bash testBOP.sh $d -m rbop --treeType completeRandom
    bash testBOP.sh $d -m rbop --treeType hierarchicalKMeans --maxLeaves 16

    # If hierarchy available
    if [ -e "data/${d}/${d}.hier" ]; then
        bash testBOP.sh $d -m rbop --treeStructure "data/${d}/${d}.hier"
        bash testBOP.sh $d -m hsmubop --treeStructure "data/${d}/${d}.hier"
    fi

    bash testBOP.sh $d -m hsmubop --treeType completeRandom
    bash testBOP.sh $d -m hsmubop --treeType hierarchicalKMeans --maxLeaves 16
    bash testBOP.sh $d -m ubop
done
