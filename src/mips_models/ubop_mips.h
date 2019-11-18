/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "br_mips.h"


class UBOPMIPS: public BRMIPS{
public:
    UBOPMIPS();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
};