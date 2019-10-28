/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "br.h"
#include "mips_index.h"


class BRMIPS: public BR{
public:
    BRMIPS();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
    void load(Args &args, std::string infile) override;

protected:
    MIPSIndex *mipsIndex;
};