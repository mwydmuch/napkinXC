/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <thread>
#include <iostream>

// Returns number of available cpus
inline int getCpuCount(){
    return std::thread::hardware_concurrency();
}

// Fowler–Noll–Vo hash
template<typename T>
uint32_t hash(T& v){
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
inline void outProgress(int state, int max){
    std::cerr << "  " << state << " / " << max << "\r";
}
