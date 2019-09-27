#!/usr/bin/env bash

#bash remap_datasets.sh

DATASETS=(VOC2006 VOC2007 CAL101 CAL256 PROTEIN2)
ARGS="-e 0.001 --norm 0 --bias 0 --weightsThreshold 0"

#rm -rf models
#rm -rf outputs
for d in "${DATASETS[@]}"; do
    echo "${d} tests..."

    bash scripts/testBOP.sh "HIERARCHICAL_PREDEFINED_${d}" -m ubopch --treeStructure "data/HIERARCHICAL_PREDEFINED_${d}/HIERARCHICAL_PREDEFINED_${d}.hier" ${ARGS}
    bash scripts/testBOP.sh "HIERARCHICAL_RANDOM_${d}" -m ubopch --treeStructure "data/HIERARCHICAL_RANDOM_${d}/HIERARCHICAL_RANDOM_${d}.hier" ${ARGS}

    bash scripts/testBOP.sh "FLAT_${d}" -m ubop ${ARGS}

    bash scripts/testBOP.sh "FLAT_${d}" -m ubopmips ${ARGS}
done
