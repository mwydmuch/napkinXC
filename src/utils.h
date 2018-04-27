/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <thread>
#include <iostream>


// Math utils
template<typename T>
inline T dotVectors(Feature* vector1, const T* vector2, int size){
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1 && f->index < size) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template<typename T>
inline T dotVectors(Feature* vector1, const std::vector<T>& vector2){
    dotVectors(vector1, vector2.data(), vector2.size());
}

template<typename T>
inline void setVector(Feature* vector1, T* vector2, int size){
    Feature* f = vector1;
    while(f->index != -1 && f->index < size){
        vector2[f->index] = f->value;
        ++f;
    }
}

template<typename T>
inline void setVector(Feature* vector1, std::vector<T>& vector2) {
    setVector(vector1, vector2.data(), vector2.size());
}

template<typename T>
inline void addVector(Feature* vector1, T* vector2, int size){
    Feature* f = vector1;
    while(f->index != -1 && f->index < size){
        vector2[f->index] += f->value;
        ++f;
    }
}

template<typename T>
inline void addVector(Feature* vector1, std::vector<T>& vector2) {
    addVector(vector1, vector2.data(), vector2.size());
}

template <typename T>
inline void unitNorm(T* data, int size){
    T norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f] * data[f];
    norm = std::sqrt(norm);
    for(int f = 0; f < size; ++f) data[f] /= norm;
}

inline void unitNorm(Feature* data, int size){
    double norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f].value * data[f].value;
    norm = std::sqrt(norm);
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
