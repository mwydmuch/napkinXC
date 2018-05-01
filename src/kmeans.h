/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <vector>
#include "types.h"

// Heuristic (Balanced) K-Means clustering
struct Assignation{
    int index;
    int value;
};

struct Distances{
    int index;
    std::vector<Feature> values;

    bool operator<(const Distances &r) const { return values[0].value < r.values[0].value; }
};

// Partition is returned via reference
void kMeans(std::vector<Assignation>* partition, SRMatrix<Feature>& pointsFeatures,
                    int centroids, double eps, bool balanced, int seed);
