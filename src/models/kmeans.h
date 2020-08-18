/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <unordered_set>
#include <vector>

#include "types.h"

// K-Means clustering with balanced option
typedef IntFeature Assignation;

struct Similarities {
    int index;
    std::vector<Feature> values;
    double sortby;

    bool operator<(const Similarities& r) const { return sortby < r.sortby; }
};

// Partition is returned via reference, calculated for cosine distance
void kmeans(std::vector<Assignation>* partition, SRMatrix<Feature>& pointsFeatures, int centroids, double eps,
            bool balanced, int seed);
