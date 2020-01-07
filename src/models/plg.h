/**
 * Copyright (c) 2020 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <unordered_set>

#include "base.h"
#include "model.h"
#include "types.h"

class UniversalHash {
public:
    UniversalHash(int a, int b) : a(a), b(b){};

    int a;
    int b;

    int hash(int value) { return a * value % b; };
};

// Probabilistic Label Graph
class PLG : public Model {
public:
    PLG();
    ~PLG() override;

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) override;
    double predictForLabel(Label label, Feature* features, Args& args) override;

    void load(Args& args, std::string infile) override;
    inline int baseForLabel(int label, int hash) {
        return (hash * layerSize) + (hashes[hash].hash(label) % layerSize);
    }

    static bool isPrime(int number);
    static int getFirstBiggerPrime(int number);

protected:
    std::vector<Base*> bases;

    int layerSize;
    std::vector<UniversalHash> hashes;
    std::vector<std::vector<int>> baseToLabels;
};
