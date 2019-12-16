/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "linear.h"

typedef int Label;
typedef int Example;
typedef feature_node DoubleFeature;
typedef DoubleFeature Feature;

struct IntFeature {
    int index;
    int value;

    bool operator<(const IntFeature& r) const { return value < r.value; }

    friend std::ostream& operator<<(std::ostream& os, const IntFeature& f) {
        os << f.index << ":" << f.value;
        return os;
    }
};

// Elastic low-level sparse row matrix, type T needs to contain int at offset 0!
template <typename T> class SRMatrix {
public:
    SRMatrix();
    ~SRMatrix();

    inline void appendRow(const std::vector<T>& row);
    void appendRow(const T* row, const int size);

    inline void replaceRow(const int index, const std::vector<T>& row);
    void replaceRow(const int index, const T* row, const int size);

    // Inefficient operation
    void appendToRow(const int index, const std::vector<T>& row);
    void appendToRow(const int index, const T* data, const int size = 1);

    // Row multiplication
    template <typename U> inline U dotRow(const int index, const std::vector<U>& vector);

    template <typename U> inline U dotRow(const int index, const U* vector);

    template <typename U> inline U dotRow(const int index, const U* vector, const int size);

    // Returns data as T**
    inline T** data() { return r.data(); }
    // inline const T** data() const { return r.data(); }

    // Returns row as T*
    // inline T* row(const int index) { return r[index]; }
    inline T* row(const int index) const { return r[index]; }

    // Returns std::vector<T*>&
    inline std::vector<T*>& allRows() { return r; }

    // Access row also by [] operator
    inline T* operator[](const int index) { return r[index]; }
    inline const T* operator[](const int index) const { return r[index]; }

    // Returns rows' sizes
    inline std::vector<int>& allSizes() { return s; }

    // Returns single row size
    inline int size(const int index) const { return s[index]; }

    // Returns size of matrix
    inline int rows() const { return m; }
    inline int cols() const { return n; }
    inline int cells() const { return c; }
    inline unsigned long long mem() { return (c + n) * sizeof(T); }

    void clear();
    void save(std::ostream& out);
    void load(std::istream& in);

private:
    int m;              // Row count
    int n;              // Col count
    int c;              // Non zero cells count
    std::vector<int> s; // Rows' sizes
    std::vector<T*> r;  // Rows

    inline T* createNewRow(const T* row, const int size);
    inline void updateN(const T* row, const int size);
};

template <typename T> SRMatrix<T>::SRMatrix() {
    m = 0;
    n = 0;
    c = 0;
}

template <typename T> SRMatrix<T>::~SRMatrix() { clear(); }

template <typename T> inline T* SRMatrix<T>::createNewRow(const T* row, const int size) {
    T* newRow = new T[size + 1];
    std::memcpy(newRow, row, size * sizeof(T));
    std::memset(&newRow[size], -1, sizeof(int)); // Add termination feature (-1)
    return newRow;
}

template <typename T> inline void SRMatrix<T>::updateN(const T* row, const int size) {
    if (size > 0) {
        int rown = *(int*)&row[size - 1] + 1;
        if (n < rown) n = rown;
    }
}

// Data should be sorted
template <typename T> inline void SRMatrix<T>::appendRow(const std::vector<T>& row) {
    appendRow(row.data(), row.size());
}

template <typename T> void SRMatrix<T>::appendRow(const T* row, const int size) {
    s.push_back(size);
    r.push_back(createNewRow(row, size));
    updateN(row, size);
    m = r.size();
    c += size;
}

// Data should be sorted
template <typename T> inline void SRMatrix<T>::replaceRow(const int index, const std::vector<T>& row) {
    replaceRow(index, row.data(), row.size());
}

template <typename T> void SRMatrix<T>::replaceRow(const int index, const T* row, const int size) {
    c += size - s[index];
    s[index] = size;
    delete[] r[index];
    r[index] = createNewRow(row, size);
    updateN(row, size);
}

template <typename T> inline void SRMatrix<T>::appendToRow(const int index, const std::vector<T>& data) {
    appendToRow(index, data.data(), data.size());
}

template <typename T> inline void SRMatrix<T>::appendToRow(const int index, const T* data, const int size) {
    int rSize = s[index];
    T* newRow = new T[s[index] + size + 1];
    std::memcpy(newRow, r[index], rSize * sizeof(T));
    std::memcpy(newRow + rSize, data, size * sizeof(T));
    std::memset(&newRow[size], -1, sizeof(int)); // Add termination feature (-1)
    delete[] r[index];
    r[index] = newRow;
    s[index] += size;
    c += size;
}

template <typename T>
template <typename U>
inline U SRMatrix<T>::dotRow(const int index, const std::vector<U>& vector) {
    return dotVectors(r[index], vector.data(), vector.size());
}

template <typename T>
template <typename U>
inline U SRMatrix<T>::dotRow(const int index, const U* vector, const int size) {
    return dotVectors(r[index], vector, size);
}

template <typename T>
template <typename U>
inline U SRMatrix<T>::dotRow(const int index, const U* vector) { // Version without size checks
    return dotVectors(r[index], vector);
}

template <typename T> void SRMatrix<T>::clear() {
    for (auto row : r) delete[] row;
    r.clear();
    s.clear();

    m = 0;
    n = 0;
    c = 0;
}

template <typename T> void SRMatrix<T>::save(std::ostream& out) {
    out << m << " " << n << "\n";
    for (int i = 0; i < m; ++i) {
        out << s[i];
        for (int j = 0; j < s[i]; ++j) out << " " << r[i][j];
        out << "\n";
    }
}

template <typename T> void SRMatrix<T>::load(std::istream& in) {
    clear();

    in.read((char*)&m, sizeof(m));
    in.read((char*)&n, sizeof(n));

    r.resize(m);
    s.resize(m);

    for (int i = 0; i < m; ++i) {
        int size;
        in.read((char*)&size, sizeof(size));
        T* newRow = new T[size + 1];
        s[i] = size;
        r[i] = newRow;
        c += size;

        for (int j = 0; j <= size; ++j) in.read((char*)&r[i][j], sizeof(T));
    }
}
