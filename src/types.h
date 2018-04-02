/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdlib>
#include <cstddef>
#include <cassert>
#include <vector>
#include "linear.h"

typedef int Label;
typedef feature_node Feature;

// Elastic sparse row matrix, type T needs to contain int at offset 0!
template <typename T>
class SRMatrix {
public:
    SRMatrix();
    ~SRMatrix();
    void appendRow(std::vector<T> row);

    T** data();
    std::vector<int>& sizes();
    int rows();
    int cols();

private:
    int m; // row count
    int n; // col count
    std::vector<int> s; // sizes
    std::vector<T *> r; // rows
};

template <typename T>
SRMatrix<T>::SRMatrix(){
    m = 0;
    n = 0;
}

template <typename T>
SRMatrix<T>::~SRMatrix(){
    for(auto row : r) delete[] row;
}

// Data should be sorted
template <typename T>
void SRMatrix<T>::appendRow(std::vector<T> row){
    s.push_back(row.size());

    T *newRow = new T[row.size() + 1];
    std::memcpy(newRow, row.data(), row.size() * sizeof(T));
    std::memset(&newRow[row.size()], -1, sizeof(T)); // Add termination feature (-1)
    r.push_back(newRow);

    if(row.size() > 0){
        int rown = *(int *)&row.back() + 1;
        if(n < rown) n = rown;
    }

    m = r.size();
}

// Returns data as T**
template <typename T>
inline T** SRMatrix<T>::data(){
    return r.data();
}

// Returns rows' sizes
template <typename T>
inline std::vector<int>& SRMatrix<T>::sizes(){
    return s;
}

template <typename T>
inline int SRMatrix<T>::rows(){
    return m;
}

template <typename T>
inline int SRMatrix<T>::cols(){
    return n;
}
