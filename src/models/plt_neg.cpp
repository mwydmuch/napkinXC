/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cassert>
#include <algorithm>
#include <vector>
#include <list>
#include <cmath>
#include <climits>

#include "plt_neg.h"

void PLTNeg::assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args){



    std::string tmpPltDir = joinPath(args.output, "tmpPlt");
    makeDir(tmpPltDir);
    PLT plt;
    plt.tree = tree;
    plt.train(labels, features, args, tmpPltDir);
    plt.load(args, tmpPltDir);

    // Positive and negative nodes
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    // Gather examples for each node
    std::cerr << "Assigning data points to nodes ...\n";
    int rows = features.rows();
    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels.row(r), labels.size(r));

        // Predict additional labels
        std::vector<Prediction> pltPrediction;
        plt.predictTopK(pltPrediction, features.row(r), args.sampleK);
        for (const auto& p : pltPrediction){
            TreeNode *n = tree->leaves[p.label];
            if(!nPositive.count(n)) nNegative.insert(n);
            else continue;
            while (n->parent) {
                n = n->parent;
                if(!nPositive.count(n)) nNegative.insert(n);
                else break;
            }
        }

        addFeatures(binLabels, binFeatures, nPositive, nNegative, features.row(r));

        nCount += nPositive.size() + nNegative.size();
        ++rCount;
    }

    remove(tmpPltDir);
}
