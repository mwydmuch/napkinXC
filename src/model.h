/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include <fstream>

#include "types.h"
#include "args.h"

struct Prediction{
    int label;
    double value; // Node's value/probability/loss

    bool operator<(const Prediction &r) const { return value < r.value; }
};

class Model{
public:
    Model();
    virtual ~Model();

    void test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args);
    virtual void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args) = 0;
    virtual void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) = 0;

    virtual void load(std::string infile) = 0;

    virtual void printInfo(){}
};

std::shared_ptr<Model> modelFactory(Args &args);
