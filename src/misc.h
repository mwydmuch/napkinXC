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

#include "log.h"
#include "types.h"

// Data utils
std::vector<Prediction> computeLabelsPriors(const SRMatrix<Label>& labels);

void computeLabelsFeaturesMatrixThread(std::vector<std::vector<Feature>>& labelsFeatures,
                                       std::vector<std::vector<int>>& labelsExamples,
                                       const SRMatrix<Label>& labels, const SRMatrix<Feature>& features,
                                       bool norm, bool weightedFeatures, int threadId, int threads);

void computeLabelsFeaturesMatrix(SRMatrix<Feature>& labelsFeatures, const SRMatrix<Label>& labels,
                                 const SRMatrix<Feature>& features, int threads = 1, bool norm = false,
                                 bool weightedFeatures = false);

// Math utils
template <typename T, typename U> inline T argMax(const UnorderedMap<T, U>& map) {
    auto pMax = std::max_element(map.begin(), map.end(), [](const std::pair<T, U>& p1, const std::pair<T, U>& p2) {
        return p1.second < p2.second;
    });
    return pMax.first;
}

template <typename T, typename U> inline T argMin(const UnorderedMap<T, U>& map) {
    auto pMin = std::min_element(map.begin(), map.end(), [](const std::pair<T, U>& p1, const std::pair<T, U>& p2) {
        return p1.second < p2.second;
    });
    return pMin.first;
}

template <typename T, typename U> inline T max(const UnorderedMap<T, U>& map) {
    auto pMax = std::max_element(map.begin(), map.end(), [](const std::pair<T, U>& p1, const std::pair<T, U>& p2) {
        return p1.first < p2.first;
    });
    return pMax.first;
}

template <typename T, typename U> inline T min(const UnorderedMap<T, U>& map) {
    auto pMin = std::min_element(map.begin(), map.end(), [](const std::pair<T, U>& p1, const std::pair<T, U>& p2) {
        return p1.first < p2.first;
    });
    return pMin.first;
}


template <typename T> inline size_t argMax(const std::vector<T>& vector) {
    return std::distance(vector.begin(), std::max_element(vector.begin(), vector.end()));
}

template <typename T> inline size_t argMin(const std::vector<T>& vector) {
    return std::distance(vector.begin(), std::min_element(vector.begin(), vector.end()));
}

// Sparse vector dot dense vector
template <typename T> inline double dotVectors(Feature* vector1, T* vector2, const size_t size) {
    double val = 0;
    for(Feature* f = vector1; f->index != -1 && f->index < size; ++f) val += f->value * vector2[f->index];
    return val;
}

template <typename T> inline double dotVectors(Feature* vector1, T* vector2) { // Version without size checks
    double val = 0;
    for(Feature* f = vector1; f->index != -1; ++f) val += f->value * vector2[f->index];
    return val;
}

template <typename T> inline double dotVectors(T* vector1, T* vector2, const size_t size) {
    double val = 0;
    for(size_t i = 0; i < size; ++i) val += vector1[i] * vector2[i];
    return val;
}

template <typename T> inline double dotVectors(Feature* vector1, T& vector2) {
    return dotVectors(vector1, vector2.data(), vector2.size());
}

template <typename T> inline double dotVectors(T& vector1, T& vector2) {
    assert(vector1.size() == vector2.size());
    return dotVectors(vector1.data(), vector2.data(), vector2.size());
}

// Sets values of a dense vector to values of a sparse vector
template <typename T> inline void setVector(Feature* vector1, T* vector2, const size_t size) {
    for(Feature* f = vector1; f->index != -1 && f->index < size; ++f) vector2[f->index] = f->value;
}

template <typename T> inline void setVector(Feature* vector1, T* vector2) { // Version without size checks
    for(Feature* f = vector1; f->index != -1; ++f) vector2[f->index] = f->value;
}

template <typename T> inline void setVector(Feature* vector1, std::vector<T>& vector2) {
    setVector(vector1, vector2.data(), vector2.size());
}


// Zeros selected values of a dense vactor
template <typename T> inline void setVectorToZeros(Feature* vector1, T* vector2, const size_t size) {
    for(Feature* f = vector1; f->index != -1 && f->index < size; ++f) vector2[f->index] = 0;
}

template <typename T>
inline void setVectorToZeros(Feature* vector1, T* vector2) { // Version without size checks
    for(Feature* f = vector1; f->index != -1; ++f) vector2[f->index] = 0;
}

template <typename T> inline void setVectorToZeros(Feature* vector1, T& vector2) {
    // setVectorToZeros(vector1, vector2.data(), vector2.size());
    setVectorToZeros(vector1, vector2.data());
}


// Add values of vector 1 to vector 2 multiplied by scalar
template <typename T> inline void addVector(T* vector1, double scalar, T* vector2, const size_t size) {
    for(int i = 0; i < size; ++i) vector2[i] += vector1[i] * scalar;
}

template <typename T> inline void addVector(T& vector1, double scalar, T& vector2) {
    addVector(vector1.data(), scalar, vector2.data(), vector2.size());
}

template <typename T> inline void addVector(T& vector1, T& vector2) {
    addVector(vector1.data(), 1.0, vector2.data(), vector2.size());
}

template <typename T> inline void addVector(const Feature* vector1, double scalar, T* vector2, const size_t size) {
    Feature* f = (Feature*)vector1;
    while (f->index != -1 && f->index < size) {
        vector2[f->index] += f->value * scalar;
        ++f;
    }
}

template <typename T> inline void addVector(const Feature* vector1, double scalar, UnorderedMap<int, T>& vector2) {
    Feature* f = (Feature*)vector1;
    while (f->index != -1) {
        vector2[f->index] += f->value * scalar;
        ++f;
    }
}

template <typename T> inline void addVector(const Feature* vector1, double scalar, T& vector2) {
    addVector(vector1, scalar, vector2.data(), vector2.size());
}

template <typename T> inline void addVector(const Feature* vector1, T& vector2) {
    addVector(vector1, 1.0, vector2.data(), vector2.size());
}


// Multiply vector by scalar
template <typename T> inline void mulVector(T* vector, double scalar, const size_t size) {
    for (int f = 0; f < size; ++f) vector[f] *= scalar;
}

template <typename T> inline void mulVector(Feature* vector, double scalar) {
    for (Feature* f = vector; f->index != -1; ++f) f->value *= scalar;
}

template <typename T> inline void mulVector(T& vector, double scalar) {
    mulVector(vector.data(), scalar, vector.size());
}

// Divide vector by scalar
inline void divVector(Feature* vector, double scalar, const size_t size) {
    for (Feature* f = vector; f->index != -1; ++f) f->value /= scalar;
}

inline void divVector(Feature* vector, double scalar) {
    for (Feature* f = vector; f->index != -1; ++f) f->value /= scalar;
}

template <typename T> inline void divVector(T* vector, double scalar, const size_t size) {
    for (int f = 0; f < size; ++f) vector[f] /= scalar;
}

template <typename T> inline void divVector(T& vector, double scalar) {
    divVector(vector.data(), scalar, vector.size());
}

// Unit norm values in container
template <typename I>
typename std::enable_if<std::is_same<typename std::iterator_traits<I>::value_type, Feature>::value, void>::type
unitNorm(I begin, I end) {
    double norm = 0;
    for (auto i = begin; i != end; ++i) norm += (*i).value * (*i).value;
    if (norm == 0) return;
    norm = std::sqrt(norm);
    for (auto i = begin; i != end; ++i) (*i).value /= norm;
}

template <typename I>
typename std::enable_if<std::is_floating_point<typename std::iterator_traits<I>::value_type>::value, void>::type
unitNorm(I begin, I end) {
    double norm = 0;
    for (auto& i = begin; i != end; ++i) norm += (*i) * (*i);
    if (norm == 0) return;
    norm = std::sqrt(norm);
    for (auto& i = begin; i != end; ++i) (*i) /= norm;
}

template <typename T> inline void unitNorm(T& cont) { unitNorm(cont.begin(), cont.end()); }

// Shift index in sparse data
template <typename I>
typename std::enable_if<std::is_same<typename std::iterator_traits<I>::value_type, Feature>::value, void>::type
shift(I begin, I end, int shift) {
    for (auto i = begin; i != end; ++i) (*i).index += shift;
}

template <typename T> inline void shift(T& cont, int shift) { shift(cont.begin(), cont.end(), shift); }


inline void threshold(std::vector<Feature>& vector, double threshold) {
    int c = 0;
    for (int i = 0; i < vector.size(); ++i)
        if (vector[i].value > threshold) {
            if (c != i) {
                vector[c].index = vector[i].index;
                vector[c].value = vector[i].value;
            }
            ++c;
        }
    vector.resize(c);
}


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
        Log(CERR) << "  " << std::round(static_cast<double>(state) / (static_cast<double>(max) / 100)) << "%\r";
}

// Splits string
std::vector<std::string> split(std::string text, char d = ',');

// String to lower
std::string toLower(std::string text);

std::string formatMem(size_t mem);

inline size_t denseSize(size_t size) { return size * sizeof(Weight); }

inline size_t mapSize(size_t size) { return size * (sizeof(int) + sizeof(int) + sizeof(Weight)); }

inline size_t sparseSize(size_t size) { return size * (sizeof(int) + sizeof(Weight)); }

// Files utils
class FileHelper {
public:
    void saveToFile(std::string outfile);
    virtual void save(std::ostream& out) = 0;
    void loadFromFile(std::string infile);
    virtual void load(std::istream& in) = 0;
};

template <typename T> inline void saveVar(std::ostream& out, T& var) { out.write((char*)&var, sizeof(var)); }

template <typename T> inline void loadVar(std::istream& in, T& var) { in.read((char*)&var, sizeof(var)); }

inline void saveVar(std::ostream& out, std::string& var) {
    size_t size = var.size();
    out.write((char*)&size, sizeof(size));
    out.write((char*)&var[0], size);
}

inline void loadVar(std::istream& in, std::string& var) {
    size_t size;
    in.read((char*)&size, sizeof(size));
    var.resize(size);
    in.read((char*)&var[0], size);
}

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
