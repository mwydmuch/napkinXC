/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "br.h"

// One against all
class OVR : public BR { // OVR is multi-class version of BR
public:
    OVR();
    double predictForLabel(Label label, Feature* features, Args& args) override;

protected:
    std::vector<Prediction> predictForAllLabels(Feature* features, Args& args) override;
};
