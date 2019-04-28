/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <tuple>
#include <algorithm>
#include <random>

#include "args.h"
#include "base.h"
#include "model.h"
#include "tree.h"
#include "types.h"


class PLT: public Model{
public:
    PLT();
    ~PLT() override;

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args) override;
    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;

    void load(std::string infile) override;

private:
    Tree* tree;
    std::vector<Base*> bases;

    void trainTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args);
};
