/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "args.h"
#include "types.h"

enum CodingType { dense, sparse, spaceOptimal };

class Base {
public:
    Base();
    ~Base();

    void train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args);
    double predict(Feature* features);

    inline bool sprase(){ return sparse; }

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile, CodingType coding = spaceOptimal);
    void load(std::istream& in, CodingType coding = spaceOptimal);

private:
    bool sparse;
    int wSize;
    int classCount;
    int firstClass;

    std::vector<double> W;
    //std::vector<Feature> sparseW;
    std::unordered_map<int, double> sparseW;

    // LibLinear's stuff
    model *M;

    // For testing/debuging
    bool useLinearPredict;
};
