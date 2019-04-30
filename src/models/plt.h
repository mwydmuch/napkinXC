/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "base.h"
#include "models/model.h"
#include "tree.h"


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
};
