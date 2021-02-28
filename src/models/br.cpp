/*
 Copyright (c) 2019-2021 by Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <list>
#include <vector>

#include "br.h"
#include "threads.h"


BR::BR() {
    type = br;
    name = "BR";
}

void BR::unload() {
    for (auto b : bases) delete b;
    bases.clear();
    bases.shrink_to_fit();
}

void BR::assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<Feature*>& binFeatures, std::vector<double>& binWeights,
                          SRMatrix<Label>& labels, SRMatrix<Feature>& features, int rStart, int rStop, Args& args){
    int rows = labels.rows();

    binWeights.resize(rows, 1);
    binFeatures.resize(rows);
    for (auto &bl: binLabels) bl.resize(rows, 0);

    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        int rSize = labels.size(r);
        auto rLabels = labels[r];

        binFeatures[r] = features[r];
        for (int i = 0; i < rSize; ++i)
            if (rLabels[i] >= rStart && rLabels[i] < rStop) binLabels[rLabels[i] - rStart][r] = 1;
    }
}

void BR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {
    int lCols = labels.cols();
    int parts = calculateNumberOfParts(labels, features, args);
    int range = lCols / parts + 1;

    assert(lCols < range * parts);
    std::vector<std::vector<double>> binLabels(range);
    std::vector<Feature*> binFeatures;
    std::vector<double> binWeights;
    std::vector<ProblemData> binProblemData;
    binWeights.reserve(range);

    std::ofstream out(joinPath(output, "weights.bin"));
    saveVar(out, lCols);

    for (int p = 0; p < parts; ++p) {
        int rStart = p * range;
        int rStop = (p + 1) * range;

        if (parts > 1)
            Log(CERR) << "Assigning labels for base estimators [" << rStart << ", " << rStop << ") (" << p + 1 << "/" << parts << ") ...\n";
        else
            Log(CERR) << "Assigning labels for base estimators ...\n";

        for (auto &bl: binLabels) std::fill(bl.begin(), bl.end(), 0);

        assignDataPoints(binLabels, binFeatures, binWeights, labels, features, rStart, rStop, args);

        unsigned long long usedMem = binFeatures.size() * ((range + 1) * sizeof(double) + sizeof(void*));
        Log(CERR) << "  Temporary data size: " << formatMem(usedMem) << "\n";

        // Train bases
        for(int i = 0; i < range; ++i) binProblemData.emplace_back(binLabels[i], binFeatures, features.cols(), binWeights);

        if(!labelsWeights.empty()) {
            Log(CERR) << "Setting inv ps weights for training ...\n";
            for (int i = 0; i < range; ++i) binProblemData[i].invPs = labelsWeights[i + rStart];
        }

        trainBases(out, binProblemData, args);

        for (auto& l : binLabels) l.clear();
        binFeatures.clear();
        binWeights.clear();
        binProblemData.clear();
    }

    out.close();
}

void BR::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    prediction = predictForAllLabels(features, args);

    if(!labelsWeights.empty())
        for(auto &p : prediction) p.value *= labelsWeights[p.label];

    if(!thresholds.empty()){
        int j = 0;
        for(int i = 0; i < prediction.size(); ++i){
            if(prediction[i].value > thresholds[i])
                prediction[j++] = prediction[i];
        }
        prediction.resize(j - 1);
    }

    sort(prediction.rbegin(), prediction.rend());
    if (args.threshold > 0) {
        int i = 0;
        while (prediction[i++].value > args.threshold);
        prediction.resize(i - 1);
    }
    if (args.topK > 0) prediction.resize(args.topK);
    prediction.shrink_to_fit();
}

std::vector<Prediction> BR::predictForAllLabels(Feature* features, Args& args) {
    std::vector<Prediction> prediction;
    prediction.reserve(bases.size());
    for (int i = 0; i < bases.size(); ++i) prediction.emplace_back(i, bases[i]->predictProbability(features));

    return prediction;
}

double BR::predictForLabel(Label label, Feature* features, Args& args) {
    return bases[label]->predictProbability(features);
}

void BR::load(Args& args, std::string infile) {
    Log(CERR) << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"), args.resume, args.loadDense);
    m = bases.size();

    loaded = true;
}

void BR::printInfo() {
    Log(COUT) << name << " additional stats:"
              << "\n  Mean # estimators per data point: " << bases.size() << "\n";
}

size_t BR::calculateNumberOfParts(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    int rows = features.rows();
    int lCols = labels.cols();
    int lCells = labels.cells();

    // Calculate required memory
    // Size of required data
    unsigned long long dataMem = labels.mem() + features.mem();
    unsigned long long tmpDataMem = 0;
    if(args.modelType == ovr && args.pickOneLabelWeighting)
        tmpDataMem = lCells * ((lCols + 1) * sizeof(double) + sizeof(void*));
    else tmpDataMem = rows * ((lCols + 1) * sizeof(double) + sizeof(void*));
    unsigned long long baseMem = 4 * args.threads * features.cols() * sizeof(double);
    unsigned long long reqMem = tmpDataMem + dataMem + baseMem;
    //Log(CERR) << "Required memory to train: " << formatMem(reqMem) << ", available memory: " << formatMem(args.memLimit) << "\n";
    Log(CERR) << "Required memory to train: " << formatMem(reqMem) << " (data: " << formatMem(dataMem)
              << ", weights: " << formatMem(baseMem) << ", tmp data: " << formatMem(tmpDataMem) << "), available memory: " << formatMem(args.memLimit) << "\n";

    size_t parts = tmpDataMem / (args.memLimit - dataMem - baseMem) + 1;
    return parts;
}
