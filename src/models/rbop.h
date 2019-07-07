/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "hsm.h"

// One against all
class RBOP: public HSM{
public:
    RBOP();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
};