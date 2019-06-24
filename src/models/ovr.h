/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "br.h"

// One against all
class OVR: public BR{
public:
    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
    void checkRow(Label* labels, Feature* feature) override;
    double predict(Label label, Feature* features, Args &args) override;

    void printInfo() override;
};
