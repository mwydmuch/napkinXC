/*
 Copyright (c) 2018-2020 by Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "basic_types.h"
#include "matrix.h"
#include "log.h"

// Data utils
std::vector<Prediction> computeLabelsPriors(const SRMatrix& labels);

void computeLabelsFeaturesMatrixThread(std::vector<std::vector<Feature>>& labelsFeatures,
                                       std::vector<std::vector<int>>& labelsExamples,
                                       const SRMatrix& labels, const SRMatrix& features,
                                       bool norm, bool weightedFeatures, int threadId, int threads);

void computeLabelsFeaturesMatrix(SRMatrix& labelsFeatures, const SRMatrix& labels,
                                 const SRMatrix& features, int threads = 1, bool norm = false,
                                 bool weightedFeatures = false);


// Other utils

// Fowler–Noll–Vo hash
template <typename T> inline uint32_t hash(T& v) {
    size_t size = sizeof(T);
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&v);
    uint32_t h = 2166136261;
    for (size_t i = 0; i < size; i++) {
        h = h ^ static_cast<int>(bytes[i]);
        h = h * 16777619;
    }
    return h;
}

// Prints progress
inline void printProgress(int state, int max) {
    if (max < 100 || state % (max / 100) == 0)
        Log(CERR) << "  " << std::round(static_cast<Real>(state) / (static_cast<Real>(max) / 100)) << "%\r";
}

// Splits string
std::vector<std::string> split(std::string text, char d = ',');

// String to lower
std::string toLower(std::string text);

// Print mem
std::string formatMem(size_t mem);

// Joins two paths
std::string joinPath(const std::string& path1, const std::string& path2);

// Checks filename
void checkFileName(const std::string& filename, bool read = true);

// Checks dirname
void checkDirName(const std::string& dirname);

// Run shell CMD
void shellCmd(const std::string& cmd);

// Create directory
void makeDir(const std::string& dirname);

// Remove file or directory
void remove(const std::string& path);
