/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <unordered_set>
#include <vector>

#include "types.h"

// (Heuristic) Balanced K-Means clustering
typedef IntFeature Assignation;

struct Distances {
    int index;
    std::vector<Feature> values;
    std::vector<Feature> differences;

    bool operator<(const Distances& r) const { return differences[0].value < r.differences[0].value; }
};

// Partition is returned via reference, calculated for cosine distance
void kMeans(std::vector<Assignation>* partition, SRMatrix<Feature>& pointsFeatures, int centroids, double eps,
            bool balanced, int seed);
