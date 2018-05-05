/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */


#include <cstdio>
#include "utils.h"

// Data utils

void computeLabelsFrequencies(std::vector<int>& labelsFreq, SRMatrix<Label>& labels){
    std::cerr << "Computing labels' frequencies ...\n";

    labelsFreq.clear();
    labelsFreq.resize(labels.cols());
    int rows = labels.rows();

    for(int r = 0; r < rows; ++r) {
        printProgress(r, rows);
        int rSize = labels.size(r);
        auto rLabels = labels.row(r);
        for (int i = 0; i < rSize; ++i) ++labelsFreq[rLabels[i]];
    }
}

// TODO: Make it parallel
void computeLabelsFeaturesMatrix(SRMatrix<Feature>& labelsFeatures, SRMatrix<Label>& labels, SRMatrix<Feature>& features){
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
                if (!tmpLabelsFeatures[rLabels[j]].count(rFeatures[i].index))
                    tmpLabelsFeatures[rLabels[j]][rFeatures[i].index] = 0;
                tmpLabelsFeatures[rLabels[j]][rFeatures[i].index] += rFeatures[i].value;
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

void computeLabelsExamples(std::vector<std::vector<double>>& labelsExamples, SRMatrix<Label>& labels){
    std::cerr << "Computing labels' examples ...\n";

    labelsExamples.clear();
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
std::string joinPath(std::string path1, std::string path2){
    char sep = '/';

    std::string joined = path1;
    if(path1[path1.size() - 1] != sep) joined += sep;
    if(path2[0] == sep) joined += path2.substr(1);
    else joined += path2;

    return(joined);
}

// Checks filename
void checkFileName(std::string filename, bool read){
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
void checkDirName(std::string dirname){
    std::string tmpFile = joinPath(dirname, ".checkTmp");
    std::ofstream out(tmpFile);
    if(!out.good()) throw "Invalid dirname: \"" + dirname +"\"!";
    std::remove(tmpFile.c_str());
}


