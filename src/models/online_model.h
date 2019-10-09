/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "model.h"


class OnlineModel: virtual public Model{
public:
    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output) final;

    virtual void init(int labelCount, Args &args) = 0;
    virtual void update(Label* labels, size_t labelsSize, Feature* features, size_t featuresSize, Args &args) = 0;
    virtual void save(Args &args, std::string output) = 0;
};
