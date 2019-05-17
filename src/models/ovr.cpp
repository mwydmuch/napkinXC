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

#include "ovr.h"
#include "threads.h"


OVR::OVR(){}

OVR::~OVR() {
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

void OVR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){

    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

//    std::cerr << "Assigning labels for base estimators ...\n";
//
//    std::vector<std::vector<double>> binLabels(lCols);
//    for(int i = 0; i < binLabels.size(); ++i)
//        binLabels[i].reserve(rows);
//
//    for(int r = 0; r < rows; ++r){
//        printProgress(r, rows);
//
//        int rSize = labels.size(r);
//        auto rLabels = labels.row(r);
//
//        for(int i = 0; i < binLabels.size(); ++i)
//            binLabels[i].push_back(0.0);
//
//        if(rSize == 1)
//            binLabels[rLabels[0]].back() = 1.0;
//        else {
//            if (rSize > 1) {
//                //std::cerr << "Encountered example with more then 1 label! OVR is multi-class classifier, use BR instead!";
//                continue;
//                //throw "OVR is multi-class classifier, encountered example with more then 1 label! Use BR instead.";
//            }
//            else if (rSize < 1){
//                std::cerr << "Example without label, skipping ...\n";
//                continue;
//            }
//        }
//    }
//
//    trainBasesWithSameFeatures(joinPath(args.output, "ovr_weights.bin"), features.cols(), binLabels, features.allRows(), args);


    std::ofstream out(joinPath(args.output, "ovr_weights.bin"));
    int size = lCols;
    out.write((char*) &size, sizeof(size));

    int parts = 16;
    int range = lCols / parts + 1;

    std::vector<std::vector<double>> binLabels(range);
    for(int i = 0; i < binLabels.size(); ++i)
        binLabels[i].reserve(rows);

    for(int p = 0; p < parts; ++p){

        std::cerr << "Assigning labels for base estimators (" << p + 1 << "/" << parts << ")...\n";

        int rStart = p * range;
        int rStop = (p + 1) * range;

        for(int r = 0; r < rows; ++r){
            printProgress(r, rows);

            int rSize = labels.size(r);
            auto rLabels = labels.row(r);

            for(int i = 0; i < binLabels.size(); ++i)
                binLabels[i].push_back(0.0);

            if(rSize == 1 && rLabels[0] >= rStart && rLabels[0] < rStop)
                binLabels[rLabels[0] - rStart].back() = 1.0;
            else {
                if (rSize > 1) {
                    //std::cerr << "Encountered example with more then 1 label! OVR is multi-class classifier, use BR instead!";
                    continue;
                    //throw "OVR is multi-class classifier, encountered example with more then 1 label! Use BR instead.";
                }
                else if (rSize < 1){
                    std::cerr << "Example without label, skipping ...\n";
                    continue;
                }
            }
        }

        trainBasesWithSameFeatures(out, features.cols(), binLabels, features.allRows(), args);

        for(int i = 0; i < binLabels.size(); ++i)
            binLabels[i].clear();
    }

    out.close();
}

void OVR::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    double sum = 0;
    for(int i = 0; i < bases.size(); ++i) {
        double value = bases[i]->predictProbability(features);
        sum += value;
        prediction.push_back({i, value});
    }

    for(auto& p : prediction)
        p.value /= sum;

    sort(prediction.rbegin(), prediction.rend());
    if(args.topK > 0) prediction.resize(args.topK);
}

void OVR::load(std::string infile){
    std::cerr << "Loading OVR model ...\n";
    bases = loadBases(joinPath(infile, "ovr_weights.bin"));
}

void OVR::printInfo(){
    std::cerr << "OVR additional stats:"
              << "\n  Mean # estimators per data point: " << bases.size()
              << "\n";
}


