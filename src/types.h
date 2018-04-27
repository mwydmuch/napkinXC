/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdlib>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <vector>
#include <fstream>
#include "linear.h"

#include <iostream>

typedef int Label;
typedef feature_node Feature;

// Elastic sparse row matrix, type T needs to contain int at offset 0!
template <typename T>
class SRMatrix {
public:
    SRMatrix();
    ~SRMatrix();
    void appendRow(const std::vector<T>& row);
    void appendRow(const T* row, const int size);

    // Row multiplication
    template <typename U>
    inline U dotRow(const int index, const std::vector<U>& vector);

    template <typename U>
    inline U dotRow(const int index, const U* vector, const int size);

    // Returns data as T**
    inline T** data(){ return r.data(); }

    // Returns row as T*
    inline T* row(const int index){ return r[index]; }

    // Access row also by [] operator
    inline T& operator[](const int index) { return r[index]; }

    // Returns rows' sizes
    inline std::vector<int>& sizes(){ return s; }

    // Returns single row size
    inline int size(const int index){ return s[index]; }

    // Returns size of matrix
    inline int rows(){ return m; }
    inline int cols(){ return n; }

    void clear();
    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile);
    void load(std::istream& in);

private:
    int m; // Row count
    int n; // Col count
    std::vector<int> s; // Rows' sizes
    std::vector<T*> r; // Rows
};

template <typename T>
SRMatrix<T>::SRMatrix(){
    m = 0;
    n = 0;
}

template <typename T>
SRMatrix<T>::~SRMatrix(){
    clear();
}

// Data should be sorted
template <typename T>
inline void SRMatrix<T>::appendRow(const std::vector<T>& row){
    appendRow(row.data(), row.size());
}

template <typename T>
void SRMatrix<T>::appendRow(const T* row, const int size){
    s.push_back(size);

    T* newRow = new T[size + 1];
    std::memcpy(newRow, row, size * sizeof(T));
    std::memset(&newRow[size], -1, sizeof(T)); // Add termination feature (-1)
    r.push_back(newRow);

    if(size > 0){
        int rown = *(int *)&row[size - 1] + 1;
        if(n < rown) n = rown;
    }

    m = r.size();
}

template <typename T>
template <typename U>
inline U SRMatrix<T>::dotRow(const int index, const std::vector<U>& vector){
    return dotVectors(r[index], vector.data(), vector.size());
}

template <typename T>
template <typename U>
inline U SRMatrix<T>::dotRow(const int index, const U* vector, const int size){
    return dotVectors(r[index], vector, size);
}

template <typename T>
void SRMatrix<T>::clear(){
    for(auto row : r) delete[] row;
    r.clear();
    s.clear();

    m = 0;
    n = 0;
}

template <typename T>
void SRMatrix<T>::save(std::string outfile){
    std::ofstream out(outfile);
    save(out);
    out.close();
}

template <typename T>
void SRMatrix<T>::save(std::ostream& out){
    out.write((char*) &m, sizeof(m));
    out.write((char*) &n, sizeof(n));
    for(int i = 0; i < m; ++i){
        out.write((char*) &s[i], sizeof(s[i]));
        for(int j = 0; j <= s[i]; ++j)
            out.write((char *) &r[i][j], sizeof(T));
    }
}

template <typename T>
void SRMatrix<T>::load(std::string infile){
    std::ifstream in(infile);
    load(in);
    in.close();
}

template <typename T>
void SRMatrix<T>::load(std::istream& in) {
    clear();

    in.read((char*) &m, sizeof(m));
    in.read((char*) &n, sizeof(n));

    r.reserve(m);
    s.reserve(m);

    for(int i = 0; i < m; ++i) {
        int size;
        in.read((char*) &size, sizeof(size));
        T* newRow = new T[size + 1];
        s.push_back(size);
        r.push_back(newRow);

        for (int j = 0; j <= size; ++j)
            in.read((char *) &r[i][j], sizeof(T));
    }
}
