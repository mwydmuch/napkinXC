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

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <queue>

#include "args.h"
#include "linear.h"
#include "save_load.h"
#include "robin_hood.h"


typedef float Weight;
typedef std::pair<int, Weight> SparseWeight;
#define UnorderedMap robin_hood::unordered_flat_map
#define UnorderedSet robin_hood::unordered_flat_set

typedef int Label;
typedef int Example;
typedef feature_node DoubleFeature;
typedef DoubleFeature Feature;

class FileHelper;

struct IntFeature {
    int index;
    int value;

    // Features are sorted by index
    bool operator<(const IntFeature& r) const { return index < r.index; }

    friend std::ostream& operator<<(std::ostream& os, const IntFeature& f) {
        os << f.index << ":" << f.value;
        return os;
    }
};

struct Prediction {
    int label;
    double value; // labels's value/probability/loss
    Prediction(){ label = 0; value = 0; }
    Prediction(int label, double value): label(label), value(value) {}

    bool operator<(const Prediction& r) const { return value < r.value; }

    friend std::ostream& operator<<(std::ostream& os, const Prediction& p) {
        os << p.label << ":" << p.value;
        return os;
    }
};


// Helping out operators
template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& pair) {
    os << pair.index << ":" << pair.value;
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[ ";
    for (auto i = vec.begin(); i != vec.end(); ++i){
        if (i != vec.begin()) os << ", ";
        os << *i;
    }
    os << " ]";
    return os;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const UnorderedMap<T, U>& map) {
    os << "{ ";
    for (auto i = map.begin(); i != map.end(); ++i){
        if (i != map.begin()) os << ", ";
        os << *i;
    }
    os << " }";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const UnorderedSet<T>& set) {
    os << "{ ";
    for (auto i = set.begin(); i != set.end(); ++i){
        if (i != set.begin()) os << ", ";
        os << *i;
    }
    os << " }";
    return os;
}


template <typename T> class TopKQueue{
public:
    TopKQueue(){
        k = 0;
    }
    explicit TopKQueue(int k): k(k){};
    ~TopKQueue() = default;

    inline bool empty(){
        return mainQueue.empty();
    }

    inline void push(T x, bool final = false){
        if(k > 0){
            if(final){
                if(finalQueue.size() < k){
                    finalQueue.push(x);
                    mainQueue.push(x);
                } else if(finalQueue.top() < x){
                    finalQueue.pop();
                    finalQueue.push(x);
                    mainQueue.push(x);
                }
            }
            else if(finalQueue.size() < k || finalQueue.top() < x) mainQueue.push(x);
        } else mainQueue.push(x);
    }

    inline void pop(){
        mainQueue.pop();
    }

    inline T top(){
        return mainQueue.top();
    }

private:
    std::priority_queue<T> mainQueue;
    std::priority_queue<T, std::vector<T>, std::greater<>> finalQueue;
    int k;
};

// Abstract vector type
template <typename T> class AbstractVector {
public:
    // Constructors
    AbstractVector(): s(0), n0(0){ };
    explicit AbstractVector(size_t s): s(s), n0(0) { };
    AbstractVector(const AbstractVector<T>& vec){
        s = vec.s;
        n0 = vec.n0;
    };

    virtual ~AbstractVector(){ };

    // Required to override
    virtual void initD() = 0;
    virtual void clearD() = 0;
    virtual void insertD(int i, T v) = 0;
    virtual void checkD() {};


    virtual AbstractVector<T>* copy() = 0;
    virtual void resize(size_t newS) {
        s = newS;
    };
    virtual void reserve(size_t maxN0) {};

    virtual inline T at(int index) const = 0;

    virtual inline T& operator[](int index) = 0;
    virtual inline const T& operator[](int index) const = 0;
    virtual void forEachD(const std::function<void(T&)>& func) = 0;
    virtual void forEachD(const std::function<void(T&)>& func) const = 0;
    virtual void forEachID(const std::function<void(const int&, T&)>& func) = 0;
    virtual void forEachID(const std::function<void(const int&, T&)>& func) const = 0;

    virtual unsigned long long mem() const = 0;

    // Basic math operations
    virtual T dot(AbstractVector<T>& vec) const{
        T val = 0;
        vec.forEachID([&](const int& i, T& v) { val += v * at(i); });
        return val;
    };

    virtual double dot(Feature* vec) const {
        double val = 0;
        for(auto f = vec; f->index != -1; ++f) val += f->value * at(f->index);
        return val;
    };

    void mul(T scalar){
        forEachD([&](T& v) { v *= scalar; });
    };

    void div(T scalar){
        forEachD([&](T& v) { v /= scalar; });
    };

    void add(T scalar){
        forEachD([&](T& v) { v += scalar; });
    };

    void add(AbstractVector& vec){
        vec.forEachID([&](const int& i, T& v) { (*this)[i] += v; });
    };

    void add(AbstractVector& vec, T scalar){
        vec.forEachID([&](const int& i, T& v) { (*this)[i] += scalar * v; });
    };

    void invert() {
        forEachD([&](T& w) { w *= -1; });
    };

    void prune(T threshold){
        forEachID([&](const int& i, Weight& w) {
            if (fabs(w) <= threshold) w = 0;
        });
        checkD();
    };

    inline size_t size() const { return s; }
    inline size_t nonZero() const { return n0; }
    size_t sparseMem() const { return n0 * (sizeof(int) + sizeof(T)); };
    size_t denseMem() const { return s * sizeof(T); };

    virtual void save(std::ostream& out) {
        checkD();
        saveVar(out, s);
        saveVar(out, n0);
        bool sparse = sparseMem() < denseMem() || s == 0; // Select more optimal coding
        saveVar(out, sparse);

        if(sparse) forEachID([&](const int& i, T& v) {
                if(v != 0) {
                    saveVar(out, i);
                    saveVar(out, v);
                }
            });
        else {
            for (int i = 0; i < s; ++i) {
                T v = at(i);
                saveVar(out, v);
            }
        }
    };

    virtual void load(std::istream& in) {
        clearD(); // Clear prev data

        // Load header
        loadVar(in, s);
        size_t n0ToLoad;
        loadVar(in, n0ToLoad);
        bool sparse;
        loadVar(in, sparse);

        // Allocate new vec
        initD();
        reserve(n0ToLoad);

        // Load and insert data
        int index;
        T value;
        if(sparse) {
            for (int i = 0; i < n0ToLoad; ++i) {
                loadVar(in, index);
                loadVar(in, value);
                insertD(index, value);
            }
        } else {
            for (int i = 0; i < s; ++i) {
                loadVar(in, value);
                if(value != 0) insertD(i, value);
            }
        }

        assert(n0 == n0ToLoad);
    };

    static void skipLoad(std::istream& in){
        size_t s, n0;
        bool sparse;
        loadVar(in, s);
        loadVar(in, n0);
        loadVar(in, sparse);
        if(sparse) in.seekg(n0 * (sizeof(int) + sizeof(T)), std::ios::cur);
        else in.seekg(s * sizeof(T), std::ios::cur);
    };

    virtual RepresentationType type() const = 0;

protected:
    size_t s;       // size
    size_t n0;      // non-zero elements
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const AbstractVector<T>& vec) {
    os << "{ ";
    vec.forEachID([&](const int& i, T& v) {
        os << "(" << i << ", " << v << ") ";
    });
    os << "}";
    return os;
}

template <typename T> class SparseVector: public AbstractVector<T> {
    using AbstractVector<T>::s;
    using AbstractVector<T>::n0;

public:
    SparseVector(): AbstractVector<T>() { initD(); };
    explicit SparseVector(size_t s): AbstractVector<T>(s) { initD(); };
    SparseVector(size_t s, size_t maxN0): AbstractVector<T>(s) {
        initD();
        reserve(maxN0);
    };
    explicit SparseVector(const AbstractVector<T>& vec): AbstractVector<T>(vec) {
        initD();
        reserve(n0);
        n0 = 0;
        vec.forEachID([&](const int& i, T& v) { insertD(i, v); });
    };
    ~SparseVector(){
        clearD();
    };

    void initD() override {
        d = nullptr;
        maxN0 = 0;
        sorted = true;
    };

    void clearD() override {
        delete[] d;
        d = nullptr;
        n0 = 0;
    };

    void insertD(int i, T v) override {
        //std::cout << n0 << " " << maxN0 << " " << i << " " << v << "\n";
        if(i >= s) s = i + 1;
        if(v != 0) {
            if(n0 >= maxN0) reserve(2 * maxN0);
            d[n0++] = {i, v};
            d[n0].first = -1;
        }
    };

    AbstractVector<T>* copy() override {
        auto newVec = new SparseVector<T>(*static_cast<AbstractVector<T>*>(this));
        return static_cast<AbstractVector<T>*>(newVec);
    };

    void reserve(size_t maxN0) override {
        auto newD = new std::pair<int, T>[maxN0 + 1]();
        this->maxN0 = maxN0;
        if(d != nullptr){
            std::copy(d, d + std::min(this->n0, maxN0), newD);
            delete[] d;
        }
        d = newD;
        n0 = std::min(this->n0, maxN0);
        d[n0].first = -1;
    };

    double dot(Feature* vec) const override {
        if(sorted) { // Also assumes that vec is sorted
            auto p = d;
            double val = 0;
            for (auto f = vec; f->index != -1; ++f) {
                while (p->first != -1 && p->first != f->index) ++p;
                if (p->first == -1) break;
                else val += f->value * p->second;
            }
            return val;
        } else return AbstractVector<T>::dot(vec);
    };

    inline T at(int index) const override {
        /*
        if(sorted){ // Binary search

        } else {

        }
         */
        // Linear search
        auto p = d;
        while (p->first != -1 && p->first != index) ++p;
        if (p->first == -1) return 0;
        else return p->second;
    };

    inline T& operator[](int index) override {
        auto p = d;
        while(p->first != -1 && p->first != index) ++p;
        return p->second;
    };

    inline const T& operator[](int index) const override {
        auto p = d;
        while(p->first != -1 && p->first != index) ++p;
        return p->second;
    };

    void forEachD(const std::function<void(T&)>& func) override {
        for(auto p = d; p->first != -1; ++p) func(p->second);
    };

    void forEachD(const std::function<void(T&)>& func) const override {
        for(auto p = d; p->first != -1; ++p) func(p->second);
    };

    void forEachID(const std::function<void(const int&, T&)>& func) override {
        for(auto p = d; p->first != -1; ++p) func(p->first, p->second);
    };

    void forEachID(const std::function<void(const int&, T&)>& func) const override {
        for(auto p = d; p->first != -1; ++p) func(p->first, p->second);
    };

    unsigned long long mem() const override { return estimateMem(s, n0); };
    static unsigned long long estimateMem(size_t s, size_t n0){
        return sizeof(SparseVector<T>) + n0 * (sizeof(int) + sizeof(T));
    }

    RepresentationType type() const override {
        return sparse;
    }

protected:
    size_t maxN0;
    size_t sorted;
    std::pair<int, T>* d; // data
};


template <typename T> class MapVector: public AbstractVector<T> {
    using AbstractVector<T>::s;
    using AbstractVector<T>::n0;

public:
    MapVector(): AbstractVector<T>() { initD(); };
    explicit MapVector(size_t s): AbstractVector<T>(s) { initD(); };
    MapVector(size_t s, size_t maxN0): AbstractVector<T>(s) {
        initD();
        reserve(maxN0);
    };
    explicit MapVector(const AbstractVector<T>& vec): AbstractVector<T>(vec) {
        initD();
        vec.forEachID([&](const int& i, T& v) { insertD(i, v); });
    };
    ~MapVector(){
        clearD();
    }

    void initD() override {
        d = new UnorderedMap<int, T>();
    };

    void clearD() override {
        delete d;
        d = nullptr;
        n0 = 0;
    };

    void insertD(int i, T v) override {
        if(i >= s) s = i + 1;
        if(v != 0) {
            (*d)[i] = v;
            n0 = d->size();
        }
    };

    void checkD() override {
        n0 = 0;
        forEachID([&](const int& i, T& v) {
            if(i >= s) s = i + 1;
            if(v != 0) ++n0;
        });
    };

    AbstractVector<T>* copy() override {
        auto newVec = new MapVector<T>(*static_cast<AbstractVector<T>*>(this));
        return static_cast<AbstractVector<T>*>(newVec);
    };

    void reserve(size_t maxN0) override {
        d->reserve(maxN0);
    };

    inline T at(int index) const override {
        auto v = d->find(index);
        if (v != d->end()) return v->second;
        else return 0;
    };
    inline T& operator[](int index) override { return (*d)[index]; }
    inline const T& operator[](int index) const override { return (*d)[index]; };

    void forEachD(const std::function<void(T&)>& func) override {
        for (auto& c : *d) func(c.second);
    };

    void forEachD(const std::function<void(T&)>& func) const override {
        for (auto& c : *d) func(c.second);
    };

    void forEachID(const std::function<void(const int&, T&)>& func) override {
        for (auto& c : *d) func(c.first, c.second);
    };

    void forEachID(const std::function<void(const int&, T&)>& func) const override {
        for (auto& c : *d) func(c.first, c.second);
    };

    unsigned long long mem() const override {
        unsigned long long mem = sizeof(MapVector);
        if(d != nullptr) mem += d->mask() * (2 * sizeof(int) + sizeof(T));
        return mem;
    };
    static unsigned long long estimateMem(size_t s, size_t n0){
        unsigned long long mem = sizeof(MapVector);
        size_t mapSize = sizeof(uint64_t);
        while(mapSize < n0) mapSize *= 2;
        mem += mapSize * (2 * sizeof(int) + sizeof(T));
        return mem;
    }

    RepresentationType type() const override {
        return map;
    }

protected:
    UnorderedMap<int, T>* d; // data
};


// Simple dense vector
template <typename T> class Vector: public AbstractVector<T> {
    using AbstractVector<T>::s;
    using AbstractVector<T>::n0;

public:
    Vector(): AbstractVector<T>() { initD(); };
    explicit Vector(size_t s): AbstractVector<T>(s) { initD(); };
    Vector(size_t s, size_t maxN0): AbstractVector<T>(s) { initD(); };
    explicit Vector(const AbstractVector<T>& vec): AbstractVector<T>(vec) {
        initD();
        vec.forEachID([&](const int& i, T& v) { d[i] = v; });
    };
    ~Vector(){
        clearD();
    }

    void initD() override {
        d = new T[s]();
    };

    void clearD() override {
        delete[] d;
        d = nullptr;
        n0 = 0;
    };

    void checkD() override {
        n0 = s;
        for(int i = 0; i < s; ++i){
            if(d[i] == 0) --n0;
        }
    };

    T dot(AbstractVector<T>& vec) const override {
        T val = 0;
        vec.forEachID([&](const int& i, T& v) { val += v * d[i]; });
        return val;
    };

    double dot(Feature* vec) const override {
        double val = 0;
        for(auto f = vec; f->index != -1; ++f) val += f->value * d[f->index];
        return val;
    };

    void insertD(int i, T v) override {
        if(d[i] == 0 && v != 0) ++n0;
        d[i] = v;
    };

    AbstractVector<T>* copy() override {
        auto newVec = new Vector<T>(*static_cast<AbstractVector<T>*>(this));
        return static_cast<AbstractVector<T>*>(newVec);
    };

    void resize(size_t newS) override {
        auto newD = new T[newS]();
        if(d != nullptr){
            std::copy(d, d + std::min(s, newS), newD);
            delete[] d;
        }
        s = newS;
        d = newD;
    };

    // Access row also by [] operator
    inline T at(int index) const override {
        if(index < s) return d[index];
        else return 0;
    };

    inline T& operator[](int index) override { return d[index]; }
    inline const T& operator[](int index) const override { return d[index]; }

    void forEachD(const std::function<void(T&)>& func) override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(d[i]);
    };

    void forEachD(const std::function<void(T&)>& func) const override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(d[i]);
    };

    void forEachID(const std::function<void(const int&, T&)>& func) override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(i, d[i]);
    };

    void forEachID(const std::function<void(const int&, T&)>& func) const override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(i, d[i]);
    };

    unsigned long long mem() const override { return estimateMem(s, n0); };
    static unsigned long long estimateMem(size_t s, size_t n0){
        return sizeof(Vector<T>) + s * sizeof(T);
    }

    RepresentationType type() const override {
        return dense;
    }

//    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
//        os << "[ ";
//        for (int i = 0; i < v.s; ++i) {
//            if (i != 0) os << ", ";
//            os << v.d[i];
//        }
//        os << " ]";
//        return os;
//    };

protected:
    T* d; // data
};


// Simple dense matrix
template <typename T> class Matrix {
public:
    Matrix();
    Matrix(size_t m, size_t n);

    // Access row also by [] operator
    inline Vector<T>& operator[](int index) { return r[index]; }
    inline const Vector<T>& operator[](int index) const { return r[index]; }

    // Returns size of matrix
    inline int rows() const { return m; }
    inline int cols() const { return n; }
    inline unsigned long long mem() const { return (m * n) * sizeof(T); }

    void save(std::ostream& out);
    void load(std::istream& in);

private:
    size_t m;                      // Row count
    size_t n;                      // Col count
    std::vector<Vector<T>> r;      // Rows
};

template <typename T> Matrix<T>::Matrix() {
    m = 0;
    n = 0;
}

template <typename T> Matrix<T>::Matrix(size_t m, size_t n): m(m), n(n) {
    r.resize(m);
    for(auto& v : r) v.resize(n);
}

template <typename T> void Matrix<T>::save(std::ostream& out) {
    out.write((char*)&m, sizeof(m));
    out.write((char*)&n, sizeof(n));
    for(auto& v : r) v.save(out);
}

template <typename T> void Matrix<T>::load(std::istream& in){
    in.read((char*)&m, sizeof(m));
    in.read((char*)&n, sizeof(n));
    r.resize(m);
    for(auto& v : r) v.load(in);
}


// Elastic low-level sparse row matrix, type T needs to contain int at offset 0!
template <typename T> class SRMatrix {
public:
    SRMatrix();
    ~SRMatrix();

    inline void appendRow(const std::vector<T>& row);
    void appendRow(const T* row, const int size);

    inline void replaceRow(int index, const std::vector<T>& row);
    void replaceRow(int index, const T* row, const int size);
    
    void appendToRow(int index, const std::vector<T>& row);
    void appendToRow(int index, const T* data, const int size = 1);

    // Returns data as T**
    inline T** data() { return r.data(); }
    // inline const T** data() const { return r.data(); }

    // Returns row as T*
    // inline T* row(int index) { return r[index]; }
    inline T* row(int index) const { return r[index]; }

    // Returns std::vector<T*>&
    inline std::vector<T*>& allRows() { return r; }

    // Access row also by [] operator
    inline T* operator[](int index) { return r[index]; }
    inline const T* operator[](int index) const { return r[index]; }

    // Compare matrices with == operator
    inline bool operator==(const SRMatrix<T>& rm){
        if(m != rm.m || n != rm.n || c != rm.c)
            return false;

        for(int i = 0; i < m; ++i){
            if(s[i] != rm.s[i])
                return false;
            //else if(std::equal(r[i], r[i] + s[i], rm.r[i]))
            else if(memcmp (r[i], rm.r[i], sizeof(T) * s[i]))
                return false;
        }
        return true;
    }

    inline bool operator!=(const SRMatrix<T>& rm){
        return !operator==(rm);
    }

    // Returns rows' sizes
    inline std::vector<int>& allSizes() { return s; }

    // Returns single row size
    inline int size(int index) const { return s[index]; }

    // Returns size of matrix
    inline int rows() const { return m; }
    inline int cols() const { return n; }
    inline int cells() const { return c; }

    // (Number of cells + number of -1 cells) * size of type T + ~size of vector<T>
    inline unsigned long long mem() { return (c + n) * sizeof(T) + m * (sizeof(int) + sizeof(T*)); }

    void clear();
    void dump(std::string outfile);
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
template <typename T> inline void SRMatrix<T>::replaceRow(int index, const std::vector<T>& row) {
    replaceRow(index, row.data(), row.size());
}

template <typename T> void SRMatrix<T>::replaceRow(int index, const T* row, const int size) {
    c += size - s[index];
    s[index] = size;
    delete[] r[index];
    r[index] = createNewRow(row, size);
    updateN(row, size);
}

template <typename T> inline void SRMatrix<T>::appendToRow(int index, const std::vector<T>& data) {
    appendToRow(index, data.data(), data.size());
}

template <typename T> inline void SRMatrix<T>::appendToRow(int index, const T* data, const int size) {
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

template <typename T> void SRMatrix<T>::clear() {
    for (auto row : r) delete[] row;
    r.clear();
    s.clear();

    m = 0;
    n = 0;
    c = 0;
}

template <typename T> void SRMatrix<T>::dump(std::string outfile) {
    std::ofstream out(outfile);

    out << m << " " << n << "\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < s[i]; ++j) out << r[i][j] << " ";
        out << "\n";
    }

    out.close();
}

template <typename T> void SRMatrix<T>::save(std::ostream& out) {
    out.write((char*)&m, sizeof(m));
    out.write((char*)&n, sizeof(n));

    for (int i = 0; i < m; ++i) {
        out.write((char*)&s[i], sizeof(s[i]));
        for (int j = 0; j <= s[i]; ++j) out.write((char*)&r[i][j], sizeof(T));
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
