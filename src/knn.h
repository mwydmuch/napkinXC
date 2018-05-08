/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <vector>
#include <unordered_set>

#include "types.h"
#include "pltree.h"

// Multi Label K-Nearest Neighbor Search
class KNN{
public:
    KNN();
    KNN(SRMatrix<Label>* pointsLabels, SRMatrix<Feature>* pointsFeatures);
    ~KNN();

    void build(const std::vector<TreeNode*>& supportedLabels, const std::vector<std::vector<Example>>& labelsPoints);
    void build(const std::vector<int>& supportedLabels, const std::vector<std::vector<Example>>& labelsPoints);
    void predict(Feature* features, int k, std::vector<Feature>& result);

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile);
    void load(std::istream& in);
    
private:
    std::unordered_set<int> labels;
    std::vector<int> points;
    SRMatrix<Label>* pointsLabels;
    SRMatrix<Feature>* pointsFeatures;
};
