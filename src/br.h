/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "args.h"
#include "base.h"
#include "model.h"
#include "types.h"


class BR: public Model{
public:
    BR();
    ~BR() override;

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args) override;
    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;

    void load(std::string infile) override;

private:
    std::vector<Base*> bases;
};
