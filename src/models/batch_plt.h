/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "plt.h"


class BatchPLT: virtual public PLT{
public:
    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output) override;
};
