/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <thread>
#include <iostream>

#include "types.h"

// Data utils
void computeLabelsFrequencies(std::vector<int>& labelsFreq, SRMatrix<Label>& labels);

void computeLabelsFeaturesMatrix(SRMatrix<Feature>& labelsFeatures, SRMatrix<Label>& labels, SRMatrix<Feature>& features);

void computeLabelsExamples(std::vector<std::vector<double>>& labelsExamples, SRMatrix<Label>& labels, SRMatrix<Feature>& features);


// Math utils

template <typename T, typename U>
inline T argMax(std::unordered_map<T, U>& map){
    auto pMax = std::max_element(map.begin(), map.end(),
                                 [](const std::pair<T, U>& p1, const std::pair<T, U>& p2)
                                 { return p1.second < p2.second; });
    return pMax.first;
}

template <typename T, typename U>
inline T argMin(std::unordered_map<T, U>& map){
    auto pMin = std::min_element(map.begin(), map.end(),
                                 [](const std::pair<T, U>& p1, const std::pair<T, U>& p2)
                                 { return p1.second < p2.second; });
    return pMin.first;
}


template <typename T>
inline size_t argMax(std::vector<T>& vector){
    return std::distance(vector.begin(), std::max_element(vector.begin(), vector.end()));
}

template <typename T>
inline size_t argMin(std::vector<T>& vector){
    return std::distance(vector.begin(), std::min_element(vector.begin(), vector.end()));
}

// Sparse vector dot dense vector
template <typename T>
inline T dotVectors(Feature* vector1, const T* vector2, int size){
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1 && f->index < size) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template <typename T>
inline T dotVectors(Feature* vector1, const std::vector<T>& vector2){
    dotVectors(vector1, vector2.data(), vector2.size());
}

// Sparse vector dot sparse vector
// TODO
inline double dotVectors(Feature* vector1, Feature* vector2){
    return 0;
}

// Sets values of a dense vector to values of a sparse vector
template <typename T>
inline void setVector(Feature* vector1, T* vector2, int size){
    Feature* f = vector1;
    while(f->index != -1 && f->index < size){
        vector2[f->index] = f->value;
        ++f;
    }
}

template <typename T>
inline void setVector(Feature* vector1, std::vector<T>& vector2) {
    setVector(vector1, vector2.data(), vector2.size());
}

// Adds values of sparse vector to dense vector
template <typename T>
inline void addVector(Feature* vector1, T* vector2, int size){
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
inline void unitNorm(T* data, int size){
    T norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f] * data[f];
    norm = std::sqrt(norm);
    if(norm == 0) norm = 1;
    for(int f = 0; f < size; ++f) data[f] /= norm;
}

inline void unitNorm(Feature* data, int size){
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
    if(state % (max / 100) == 0) std::cerr << "  " << state / (max / 100) << "%\r";
}

// Files utils

// Joins two paths
std::string joinPath(std::string path1, std::string path2);

// Checks filename
void checkFileName(std::string filename, bool read = true);

// Checks dirname
void checkDirName(std::string dirname);
