/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "hsm.h"


class UBOPHSM: public HSM{
public:
    UBOPHSM();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
};