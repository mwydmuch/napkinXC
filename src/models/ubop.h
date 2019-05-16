/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "models/ovr.h"


class UBOP: public OVR{
public:
    UBOP();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
};