#!/usr/bin/env bash

#bash remap_datasets.sh

DATASETS=(VOC2006 VOC2007 CAL101 CAL256 PROTEIN2)
#DATASETS=(VOC2006 VOC2007)
ARGS="--threshold 0 -e 0.0001 --solver L2R_LR"
ARGS="--threshold 0 -e 0.001"

for d in "${DATASETS[@]}"; do
    echo "${d} tests..."

    bash testBOP.sh "HIERARCHICAL_PREDEFINED_${d}" -m rbop --treeStructure "data/HIERARCHICAL_PREDEFINED_${d}/HIERARCHICAL_PREDEFINED_${d}.hier" ${ARGS}
    bash testBOP.sh "HIERARCHICAL_RANDOM_${d}" -m rbop --treeStructure "data/HIERARCHICAL_RANDOM_${d}/HIERARCHICAL_RANDOM_${d}.hier" ${ARGS}
    bash testBOP_with_eps.sh "HIERARCHICAL_PREDEFINED_${d}" -m rbop --treeStructure "data/HIERARCHICAL_PREDEFINED_${d}/HIERARCHICAL_PREDEFINED_${d}.hier" ${ARGS}
    bash testBOP_with_eps.sh "HIERARCHICAL_RANDOM_${d}" -m rbop --treeStructure "data/HIERARCHICAL_RANDOM_${d}/HIERARCHICAL_RANDOM_${d}.hier" ${ARGS}

    bash testBOP.sh "HIERARCHICAL_PREDEFINED_${d}" -m hsmubop --treeStructure "data/HIERARCHICAL_PREDEFINED_${d}/HIERARCHICAL_PREDEFINED_${d}.hier" ${ARGS}
    bash testBOP.sh "HIERARCHICAL_RANDOM_${d}" -m hsmubop --treeStructure "data/HIERARCHICAL_RANDOM_${d}/HIERARCHICAL_RANDOM_${d}.hier" ${ARGS}

    bash testBOP.sh "FLAT_${d}" -m ubop ${ARGS}
done
