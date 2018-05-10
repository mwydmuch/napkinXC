/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <unordered_map>

#include "knn.h"
#include "utils.h"

// Multi Label K-Nearest Neighbor Search
KNN::KNN(){}

KNN::KNN(SRMatrix<Label>* labels, SRMatrix<Feature>* features): pointsLabels(labels), pointsFeatures(features) {}

KNN::~KNN(){}

void KNN::build(const std::vector<TreeNode*>& supportedLabels, const std::vector<std::vector<Example>>& labelsPoints){
    // Build list of points
    labels.clear();
    points.clear();
    std::unordered_set<int> seenPoints;
    for(const auto& l : supportedLabels){
        labels.insert(l->label);
        for(const auto& p : labelsPoints[l->label])
            if(!seenPoints.count(p)) {
                points.push_back(p);
                seenPoints.insert(p);
            }
    }
}

void KNN::build(const std::vector<int>& supportedLabels, const std::vector<std::vector<Example>>& labelsPoints){
    // Build list of points
    labels.clear();
    points.clear();
    std::unordered_set<int> seenPoints;
    for(const auto& l : supportedLabels){
        labels.insert(l);
        for(const auto& p : labelsPoints[l])
            if(!seenPoints.count(p)) {
                points.push_back(p);
                seenPoints.insert(p);
            }
    }
}

void KNN::predict(Feature* features, int k, std::vector<Feature>& result){
    result.clear();
    if(points.empty()) return;

    assert(pointsLabels != nullptr);
    assert(pointsFeatures != nullptr);

    k = std::min(k, static_cast<int>(points.size()));

    // Turn query's sparse vector to dense
    std::vector<double> denseFeatures(pointsFeatures->cols());
    setVector(features, denseFeatures);

    // Calculate distances and select k nearest
    std::vector<Feature> distances(points.size());
    for(int i = 0; i < points.size(); ++i){
        distances[i].index = i;
        distances[i].value = pointsFeatures->dotRow(i, denseFeatures);
    }

    std::sort(distances.begin(), distances.end());

    std::unordered_map<int, double> labelsValues;
    int sumOfSimilarities = 0;
    for(int i = 0; i < k; ++i){
        int pIndex = distances[i].index;
        double pSimilarity = 1.0 - distances[i].value;
        sumOfSimilarities += pSimilarity;
        int pSize = pointsLabels->size(pIndex);
        auto pLabels = pointsLabels->row(pIndex);

        // Simple version
        //for(int j = 0; j < pSize; ++j) ++labelsValues[pLabels[j]];

        // Probability based on similarity
        for(int j = 0; j < pSize; ++j) labelsValues[pLabels[j]] += pSimilarity;
    }

    // Calculate posterior probabilities
    for(const auto& l : labelsValues)
        if(labels.count(l.first)) result.push_back({l.first, static_cast<double>(l.second) / k / sumOfSimilarities});
}

void KNN::save(std::string outfile){
    std::ofstream out(outfile);
    save(out);
    out.close();
}

void KNN::save(std::ostream& out){
    size_t size = labels.size();
    out.write((char*) &size, sizeof(size));
    for(const auto& l : labels)
        out.write((char*) &l, sizeof(l));

    size = points.size();
    out.write((char*) &size, sizeof(size));
    for(const auto& p : points)
        out.write((char*) &p, sizeof(p));

    //std::cerr << "  Saved KNN: labels: " << labels.size() << ", points: " << points.size() << "\n";
}

void KNN::load(std::string infile){
    std::ifstream in(infile);
    load(in);
    in.close();
}

void KNN::load(std::istream& in){
    size_t size;
    int value;
    in.read((char*) &size, sizeof(size));
    labels.clear();
    for(int i = 0; i < size; ++i){
        in.read((char*) &value, sizeof(value));
        labels.insert(value);
    }

    in.read((char*) &size, sizeof(size));
    points.resize(size);
    for(auto& p : points)
        in.read((char*) &p, sizeof(p));

    //std::cerr << "  Loaded KNN: labels: " << labels.size() << ", points: " << points.size() << "\n";
}
