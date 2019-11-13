/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "batch_plt.h"


void BatchPLT::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output){

    // Create tree
    if(!tree) {
        tree = new Tree();
        tree->buildTreeStructure(labels, features, args);
    }
    m = tree->numberOfLeaves();

    std::cerr << "Training tree ...\n";

    // Check data
    assert(features.rows() == labels.rows());
    assert(tree->k >= labels.cols());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(tree->t);
    std::vector<std::vector<Feature*>> binFeatures(tree->t);

    assignDataPoints(binLabels, binFeatures, labels, features, args);
    trainBases(joinPath(output, "weights.bin"), features.cols(), binLabels, binFeatures, args);

    // Save tree
    tree->saveToFile(joinPath(output, "tree.bin"));

    // Save tree structure
    tree->saveTreeStructure(joinPath(output, "tree.txt"));
}
