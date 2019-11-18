/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "batch_plt.h"


class PLTNeg: public BatchPLT{
public:
    void assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                          SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args) override;
};
