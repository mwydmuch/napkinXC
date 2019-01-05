/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */


#include <cstdio>
#include "utils.h"

// Data utils

void computeLabelsFrequencies(std::vector<Frequency>& labelsFreq, const SRMatrix<Label>& labels){
    std::cerr << "Computing labels' frequencies ...\n";

    labelsFreq.clear();
    labelsFreq.resize(labels.cols());
    for(int i = 0; i < labelsFreq.size(); ++i) {
        labelsFreq[i].index = i;
        labelsFreq[i].value = 0;
    }
    int rows = labels.rows();

    for(int r = 0; r < rows; ++r) {
        printProgress(r, rows);
        int rSize = labels.size(r);
        auto rLabels = labels.row(r);
        for (int i = 0; i < rSize; ++i) ++labelsFreq[rLabels[i]].value;
    }
}

void computeLabelsPrior(std::vector<Probability>& labelsProb, const SRMatrix<Label>& labels){
    std::cerr << "Computing labels' probabilities ...\n";

    std::vector<Frequency> labelsFreq;
    computeLabelsFrequencies(labelsFreq, labels);

    labelsProb.clear();
    labelsProb.resize(labels.cols());
    for(int i = 0; i < labelsFreq.size(); ++i) {
        labelsProb[i].index = i;
        labelsProb[i].value = static_cast<double>(labelsFreq[i].value) / labels.rows();
    }
}

// TODO: Make it work in parallel
void computeLabelsFeaturesMatrix(SRMatrix<Feature>& labelsFeatures, const SRMatrix<Label>& labels,
                                 const SRMatrix<Feature>& features, bool weightedFeatures){
    std::cerr << "Computing labels' features matrix ...\n";

    std::vector<std::unordered_map<int, double>> tmpLabelsFeatures(labels.cols());

    int rows = features.rows();
    assert(rows == labels.rows());

    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);
        int rFeaturesSize = features.size(r);
        int rLabelsSize = labels.size(r);
        auto rFeatures = features.row(r);
        auto rLabels = labels.row(r);
        for (int i = 0; i < rFeaturesSize; ++i){
            for (int j = 0; j < rLabelsSize; ++j){
                auto f = tmpLabelsFeatures[rLabels[j]].find(rFeatures[i].index);
                auto v = rFeatures[i].value;
                if(weightedFeatures) v /= rLabelsSize;
                if(f == tmpLabelsFeatures[rLabels[j]].end()) tmpLabelsFeatures[rLabels[j]][rFeatures[i].index] = v;
                else (*f).second += v;
            }
        }
    }

    for(int l = 0; l < labels.cols(); ++l){
        std::vector<Feature> labelFeatures;
        for(const auto& f : tmpLabelsFeatures[l])
            labelFeatures.push_back({f.first, f.second});
        std::sort(labelFeatures.begin(), labelFeatures.end());
        unitNorm(labelFeatures);
        labelsFeatures.appendRow(labelFeatures);
    }
}

void computeLabelsExamples(std::vector<std::vector<Example>>& labelsExamples, const SRMatrix<Label>& labels){
    std::cerr << "Computing labels' examples ...\n";

    labelsExamples.clear();
    labelsExamples.resize(labels.cols());
    int rows = labels.rows();

    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);
        int rSize = labels.size(r);
        auto rLabels = labels.row(r);
        for (int i = 0; i < rSize; ++i) labelsExamples[rLabels[i]].push_back(r);
    }
}

// Files utils

// Joins two paths
std::string joinPath(const std::string& path1, const std::string& path2){
    char sep = '/';

    std::string joined = path1;
    if(path1[path1.size() - 1] != sep) joined += sep;
    if(path2[0] == sep) joined += path2.substr(1);
    else joined += path2;

    return(joined);
}

// Checks filename
void checkFileName(const std::string& filename, bool read){
    bool valid;
    if(read) {
        std::ifstream in(filename);
        valid = in.good();
    } else {
        std::ofstream out(filename);
        valid = out.good();
    }
    if (!valid) throw "Invalid filename: \"" + filename +"\"!";
}

// Checks dirname
void checkDirName(const std::string& dirname){
    std::string tmpFile = joinPath(dirname, ".checkTmp");
    std::ofstream out(tmpFile);
    if(!out.good()) throw "Invalid dirname: \"" + dirname +"\"!";
    std::remove(tmpFile.c_str());
}


