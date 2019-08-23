/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "plt.h"


class BRPLTNeg: public Model{
public:
    BRPLTNeg();
    ~BRPLTNeg() override;

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
    double predictForLabel(Label label, Feature* features, Args &args) override;

    void load(Args &args, std::string infile) override;

protected:
    std::vector<Base*> bases;
    PLT* plt;
};
