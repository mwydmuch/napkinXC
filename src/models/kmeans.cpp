/**
 * Copyright (c) 2018-2020 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <climits>
#include <cmath>
#include <random>

#include "kmeans.h"
#include "misc.h"

// (Heuristic) Balanced K-Means clustering
// Partition is returned via reference, calculated for cosine distance
void kMeans(std::vector<Assignation>* partition, SRMatrix<Feature>& pointsFeatures, int centroids, double eps,
            bool balanced, int seed) {

    std::default_random_engine rng(seed);

    int points = partition->size();
    int features = pointsFeatures.cols();

    // if(balanced) std::cerr << "Balanced K-Means ...\n  Partition: " << partition->size() << ", centroids: " <<
    // centroids << "\n"; else std::cerr << "K-Means ...\n  Partition: " << partition->size() << ", centroids: " <<
    // centroids << "\n";

    int maxPartitionSize = points - 1, maxWithOneMore = 0;
    if (balanced) {
        maxPartitionSize = points / centroids;
        maxWithOneMore = points % centroids;
        assert(centroids * maxPartitionSize + maxWithOneMore == partition->size());
    }

    // Init centroids
    std::vector<std::vector<double>> centroidsFeatures(centroids);
    for (int i = 0; i < centroids; ++i) {
        centroidsFeatures[i].resize(features, 0);
        std::uniform_int_distribution<int> dist(0, points);
        setVector(pointsFeatures.row(dist(rng)), centroidsFeatures[i]);
    }

    double oldCos = INT_MIN, newCos = -1;

    std::vector<Distances> distances(points);
    for (int i = 0; i < points; ++i) {
        distances[i].values.resize(centroids);
        distances[i].differences.resize(centroids);
    }

    while (newCos - oldCos >= eps) {

        oldCos = newCos;
        newCos = 0;

        if(centroids == 2){ // Faster version for 2-means

            // Calculate distances to centroids
            for (int i = 0; i < points; ++i) {
                distances[i].index = i;
                for (int j = 0; j < centroids; ++j) {
                    distances[i].values[j].index = j;
                    distances[i].values[j].value = dotVectors(pointsFeatures[(*partition)[i].index], centroidsFeatures[j]);
                }
                distances[i].differences[0].value = distances[i].values[0].value - distances[i].values[1].value;
            }

            // Assign points to centroids and calculate new loss
            std::sort(distances.begin(), distances.end());

            for (int i = 0; i < points; ++i) {
                int cIndex;
                if(balanced) cIndex = (i < (maxPartitionSize + maxWithOneMore)) ? 0 : 1;
                else cIndex = (distances[i].differences[0].value <= 0) ? 0 : 1;
                (*partition)[distances[i].index].value = cIndex;
                newCos += distances[i].values[cIndex].value;
            }
        } else {
            std::vector<int> centroidsSizes(centroids, 0);

            for (int i = 0; i < points; ++i) {
                distances[i].index = i;
                for (int j = 0; j < centroids; ++j) {
                    distances[i].values[j].index = j;
                    distances[i].values[j].value = dotVectors(pointsFeatures[(*partition)[i].index], centroidsFeatures[j]);
                }

                std::sort(distances[i].values.begin(), distances[i].values.end());

                for (int j = 0; j < centroids - 1; ++j)
                    distances[i].differences[j].value = distances[i].values[j].value;
            }

            // Assign points to centroids and calculate new loss
            std::sort(distances.begin(), distances.end());

            for (int i = 0; i < points; ++i) {
                for (int j = 0; j < centroids; ++j) {
                    int cIndex = distances[i].values[j].index;
                    int lIndex = distances[i].index;

                    if (centroidsSizes[cIndex] <= maxPartitionSize ||
                        (centroidsSizes[cIndex] <= maxPartitionSize + 1 && maxWithOneMore > 0)) {
                        if (centroidsSizes[cIndex] > maxPartitionSize) --maxWithOneMore;
                        (*partition)[lIndex].value = cIndex;
                        ++centroidsSizes[cIndex];
                        newCos += distances[i].values[j].value;
                        break;
                    }
                }
            }
        }

        newCos /= points;

        // Update centroids
        for (auto& c : centroidsFeatures) std::fill(c.begin(), c.end(), 0);
        for (auto& p : (*partition)) addVector(pointsFeatures[p.index], centroidsFeatures[p.value]);
        for (auto& c : centroidsFeatures) unitNorm(c);
    }
}
