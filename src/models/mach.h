/**
 * Copyright (c) 2020 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <unordered_set>

#include "base.h"
#include "basic_types.h"
#include "model.h"

class UniversalHash {
public:
    UniversalHash(int a, int b) : a(a), b(b){};

    int a;
    int b;

    int hash(int value) { return a * value % b; };
};

// Merged-Averaged Classifiers via Hashing
class MACH : public Model {
public:
    MACH();
    ~MACH() override;

    void train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) override;
    Real predictForLabel(Label label, SparseVector& features, Args& args) override;

    void load(Args& args, std::string infile) override;
    inline int baseForLabel(int label, int hash) {
        return (hash * bucketCount) + (hashes[hash].hash(label) % bucketCount);
    }

    static bool isPrime(int number);
    static int getFirstBiggerPrime(int number);

protected:
    std::vector<Base*> bases;

    int bucketCount; // B
    std::vector<UniversalHash> hashes; // of size R
    std::vector<std::vector<int>> baseToLabels;
};
