/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <algorithm>

#include "types.h"


typedef IntFeature Frequency;
typedef DoubleFeature Probability;

// Data utils
void computeLabelsFrequencies(std::vector<Frequency>& labelsFreq, const SRMatrix<Label>& labels);

void computeLabelsPrior(std::vector<Probability>& labelsProb, const SRMatrix<Label>& labels);

void computeLabelsFeaturesMatrix(SRMatrix<Feature>& labelsFeatures, const SRMatrix<Label>& labels, const SRMatrix<Feature>& features);

void computeLabelsExamples(std::vector<std::vector<Example>>& labelsFeatures, const SRMatrix<Label>& labels);


// Math utils

template <typename T, typename U>
inline T argMax(const std::unordered_map<T, U>& map){
    auto pMax = std::max_element(map.begin(), map.end(),
                                 [](const std::pair<T, U>& p1, const std::pair<T, U>& p2)
                                 { return p1.second < p2.second; });
    return pMax.first;
}

template <typename T, typename U>
inline T argMin(const std::unordered_map<T, U>& map){
    auto pMin = std::min_element(map.begin(), map.end(),
                                 [](const std::pair<T, U>& p1, const std::pair<T, U>& p2)
                                 { return p1.second < p2.second; });
    return pMin.first;
}


template <typename T>
inline size_t argMax(const std::vector<T>& vector){
    return std::distance(vector.begin(), std::max_element(vector.begin(), vector.end()));
}

template <typename T>
inline size_t argMin(const std::vector<T>& vector){
    return std::distance(vector.begin(), std::min_element(vector.begin(), vector.end()));
}

// Sparse vector dot dense vector
template <typename T>
inline T dotVectors(Feature* vector1, const T* vector2, const int& size){
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1 && f->index < size) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template <typename T>
inline T dotVectors(Feature* vector1, const T* vector2){ // Version without size checks
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template <typename T>
inline T dotVectors(Feature* vector1, const std::vector<T>& vector2){
    //dotVectors(vector1, vector2.data(), vector2.size());
    dotVectors(vector1, vector2.data());
}

// Sets values of a dense vector to values of a sparse vector
template <typename T>
inline void setVector(Feature* vector1, T* vector2, size_t size, int shift = 0){
    Feature* f = vector1;
    while(f->index != -1 && f->index + shift < size){
        vector2[f->index + shift] = f->value;
        ++f;
    }
}

template <typename T>
inline void setVector(Feature* vector1, T* vector2, int shift = 0){ // Version without size checks
    Feature* f = vector1;
    while(f->index != -1){
        vector2[f->index + shift] = f->value;
        ++f;
    }
}

template <typename T>
inline void setVector(Feature* vector1, std::vector<T>& vector2, int shift = 0) {
    //setVector(vector1, vector2.data(), vector2.size(), shift);
    setVector(vector1, vector2.data(), shift);
}

// Zeros selected values of a dense vactor
template <typename T>
inline void setVectorToZeros(Feature* vector1, T* vector2, size_t size, int shift = 0){
    Feature* f = vector1;
    while(f->index != -1 && f->index + shift < size){
        vector2[f->index + shift] = 0;
        ++f;
    }
}

template <typename T>
inline void setVectorToZeros(Feature* vector1, T* vector2, int shift = 0){ // Version without size checks
    Feature* f = vector1;
    while(f->index != -1){
        vector2[f->index + shift] = 0;
        ++f;
    }
}

template <typename T>
inline void setVectorToZeros(Feature* vector1, std::vector<T>& vector2, int shift = 0) {
    //setVectorToZeros(vector1, vector2.data(), vector2.size());
    setVectorToZeros(vector1, vector2.data(), shift);
}

// Adds values of sparse vector to dense vector
template <typename T>
inline void addVector(Feature* vector1, T* vector2, size_t size){
    Feature* f = vector1;
    while(f->index != -1 && f->index < size){
        vector2[f->index] += f->value;
        ++f;
    }
}

template <typename T>
inline void addVector(Feature* vector1, std::vector<T>& vector2) {
    addVector(vector1, vector2.data(), vector2.size());
}

// Unit norm
template <typename T>
inline void unitNorm(T* data, size_t size){
    T norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f] * data[f];
    norm = std::sqrt(norm);
    if(norm == 0) norm = 1;
    for(int f = 0; f < size; ++f) data[f] /= norm;
}

inline void unitNorm(Feature* data, size_t size){
    double norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f].value * data[f].value;
    norm = std::sqrt(norm);
    if(norm == 0) norm = 1;
    for(int f = 0; f < size; ++f) data[f].value /= norm;
}

template <typename T>
inline void unitNorm(std::vector<T>& vector){
    unitNorm(vector.data(), vector.size());
}


// Other utils

// Fowler–Noll–Vo hash
template <typename T>
inline uint32_t hash(T& v){
    size_t size = sizeof(T);
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&v);
    uint32_t h = 2166136261;
    for (size_t i = 0; i < size; i++) {
        h = h ^ static_cast<int>(bytes[i]);
        h = h * 16777619;
    }
    return h;
}

// Returns number of available cpus
inline int getCpuCount(){
    return std::thread::hardware_concurrency();
}

// Prints progress
inline void printProgress(int state, int max){
    //std::cerr << "  " << state << " / " << max << "\r";
    if(max > 100 && state % (max / 100) == 0) std::cerr << "  " << state / (max / 100) << "%\r";
}

// Files utils

// Joins two paths
std::string joinPath(const std::string& path1, const std::string& path2);

// Checks filename
void checkFileName(const std::string& filename, bool read = true);

// Checks dirname
void checkDirName(const std::string& dirname);
