/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "model.h"


class OnlineModel: public Model{
public:
    void trainOnline(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output);
    virtual void update(Label* labels, size_t labelsSize, Feature* features, size_t featuresSize, Args &args) = 0;
    virtual void save(Args &args, std::string output) = 0;
};

std::shared_ptr<Model> modelFactory(Args &args);
