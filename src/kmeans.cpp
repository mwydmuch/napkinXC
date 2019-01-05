/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <climits>

#include "kmeans.h"
#include "utils.h"

// (Heuristic) Balanced K-Means clustering
// Partition is returned via reference, calculated for cosine distance
void kMeans(std::vector<Assignation>* partition, SRMatrix<Feature>& pointsFeatures,
                    int centroids, double eps, bool balanced, int seed){

    std::default_random_engine rng(seed);

    int points = partition->size();
    int features = pointsFeatures.cols();

    //if(balanced) std::cerr << "Balanced K-Means ...\n  Partition: " << partition->size() << ", centroids: " << centroids << "\n";
    //else std::cerr << "K-Means ...\n  Partition: " << partition->size() << ", centroids: " << centroids << "\n";

    int maxPartitionSize = points - 1, maxWithOneMore = 0;
    if(balanced){
        maxPartitionSize = points / centroids;
        maxWithOneMore = points % centroids;
        assert(centroids * maxPartitionSize + maxWithOneMore == partition->size());
    }

    // Test split
    /*
    for(int i = 0; i < partition->size(); ++i)
        partition->at(i).value = i / maxPartitionSize;
    */

    // Init centroids
    std::vector<std::vector<double>> centroidsFeatures(centroids);
    for(int i = 0; i < centroids; ++i) {
        centroidsFeatures[i].resize(features, 0);
        std::uniform_int_distribution<int> dist(0, points);
        setVector(pointsFeatures.row(dist(rng)), centroidsFeatures[i]);
    }

    double oldCos = INT_MIN, newCos = -1;

    std::vector<Distances> distances(points);
    for(int i=0; i < points; ++i ) distances[i].values.resize(centroids);

    while(newCos - oldCos >= eps){
        std::vector<int> centroidsSizes(centroids, 0);

        // Calculate distances to centroids
        for(int i = 0; i < points; ++i) {
            distances[i].index = i;
            double maxDist = INT_MIN;
            for(int j = 0; j < centroids; ++j) {
                distances[i].values[j].index = j;
                distances[i].values[j].value = pointsFeatures.dotRow((*partition)[i].index, centroidsFeatures[j]);
                if(distances[i].values[j].value > maxDist) maxDist = distances[i].values[j].value;
            }

            for(int j = 0; j < centroids; ++j)
                distances[i].values[j].value -= maxDist;

            std::sort(distances[i].values.begin(), distances[i].values.end());
        }

        // Assign points to centroids and calculate new loss
        oldCos = newCos;
        newCos = 0;

        std::sort(distances.begin(), distances.end());

        for(int i = 0; i < points; ++i){
            for(int j = 0; j < centroids; ++j){
                int cIndex = distances[i].values[j].index;
                int lIndex = distances[i].index;

                if(centroidsSizes[cIndex] <= maxPartitionSize || (centroidsSizes[cIndex] <= maxPartitionSize + 1 && maxWithOneMore > 0)) {
                    if(centroidsSizes[cIndex] > maxPartitionSize) --maxWithOneMore;
                    (*partition)[lIndex].value = cIndex;
                    ++centroidsSizes[cIndex];
                    newCos += distances[i].values[j].value;
                    break;
                }
            }
        }

        newCos /= points;

        // Update centroids
        for(int i = 0; i < centroids; ++i)
            std::fill(centroidsFeatures[i].begin(), centroidsFeatures[i].end(), 0);

        for(int i = 0; i < points; ++i){
            int lIndex = (*partition)[i].index;
            int lCentroid = (*partition)[i].value;
            addVector(pointsFeatures.row(lIndex), centroidsFeatures[lCentroid]);
        }

        // Norm new centroids
        for(int i = 0; i < centroids; ++i)
            unitNorm(centroidsFeatures[i]);
    }
}
