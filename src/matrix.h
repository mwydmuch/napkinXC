/*
 Copyright (c) 2018-2021 by Marek Wydmuch

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

#include "vector.h"

// Simple row ordered matrix
template <typename T> class RMatrix {
public:
    RMatrix(): RMatrix(0, 0) {}
    RMatrix(size_t m, size_t n): m(m), n(n) {
        r.resize(m);
        for(auto &row : r) row.resize(n);
    }

    template<typename U>
    void appendRow(const U& vec, bool sorted = true) {
        T& row = r.emplace_back(vec, sorted);
        m = r.size();
        if(row.size() > n) n = row.size();
    }

    // Access row also by [] operator
    inline T& operator[](int index) { return r[index]; }
    inline const T& operator[](int index) const { return r[index]; }

    // Returns size of matrix
    inline int rows() const { return m; }
    inline int cols() const { return n; }
    inline int cells() const {
        int totalN0 = 0; // "Cells" / non-zero elements
        for(auto &vec : r) totalN0 += vec.nonZero();
        return totalN0;
    }
    inline unsigned long long mem() const {
        int totalMem = 0; // Mem
        for(auto &vec : r) totalMem += vec.mem();
        return totalMem;
    }
    inline int size(int index) { return r[index].nonZero(); }

    void save(std::ofstream& out) {
        out.write((char*)&m, sizeof(m));
        out.write((char*)&n, sizeof(n));
        for(auto& v : r) v.save(out);
    }

    void load(std::ifstream& in){
        in.read((char*)&m, sizeof(m));
        in.read((char*)&n, sizeof(n));
        r.resize(m);
        for(auto& v : r) v.load(in);
    }

    // TODO: Improve iterators
    T* begin() { return r.data(); }
    T* end() { return r.data() + r.size(); }

private:
    size_t m;              // Row count
    size_t n;              // Col count
    std::vector<T> r;      // Rows data
};

typedef RMatrix<Vector> Matrix;
typedef RMatrix<MapVector> MRMatrix;
typedef RMatrix<SparseVector> SRMatrix;
