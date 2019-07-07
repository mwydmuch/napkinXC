#!/usr/bin/env bash

#bash remap_datasets.sh

DATASETS=(LSHTC1 Dmoz)
ARGS="-e 0.001 --norm 0"

ARGS="-e 0.001"
for d in "${DATASETS[@]}"; do
    echo "${d} tests..."

    bash testBOP.sh $d -m rbop --treeType completeRandom ${ARGS}
    bash testBOP.sh $d -m rbop --treeType hierarchicalKMeans --maxLeaves 16 ${ARGS}

    # If hierarchy available
    if [ -e "data/${d}/${d}.hier" ]; then
        bash testBOP.sh $d -m rbop --treeStructure "data/${d}/${d}.hier" ${ARGS}
        bash testBOP.sh $d -m ubopch --treeStructure "data/${d}/${d}.hier" ${ARGS}
    fi

    bash testBOP.sh $d -m ubopch --treeType completeRandom ${ARGS}
    bash testBOP.sh $d -m ubopch --treeType hierarchicalKMeans --maxLeaves 16 ${ARGS}
    bash testBOP.sh $d -m ubop ${ARGS}
done
