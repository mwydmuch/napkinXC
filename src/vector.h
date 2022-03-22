/*
 Copyright (c) 2021 by Marek Wydmuch

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

#include "basic_types.h"
#include "enums.h"

#include <cmath>

// Basic vector operations

// Sparse vector dot dense vector
template <typename T> inline Real dotVectors(Feature* vector1, T* vector2, const size_t size) {
    Real val = 0;
    for(Feature* f = vector1; f->index != -1 && f->index < size; ++f) val += f->value * vector2[f->index];
    return val;
}

template <typename T> inline Real dotVectors(Feature* vector1, T* vector2) { // Version without size checks
    Real val = 0;
    for(Feature* f = vector1; f->index != -1; ++f) val += f->value * vector2[f->index];
    return val;
}

template <typename T> inline Real dotVectors(T* vector1, T* vector2, const size_t size) {
    Real val = 0;
    for(size_t i = 0; i < size; ++i) val += vector1[i] * vector2[i];
    return val;
}

template <typename T> inline Real dotVectors(T& vector1, T& vector2) {
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
template <typename T> inline void addVector(T* vector1, Real scalar, T* vector2, const size_t size) {
    for(int i = 0; i < size; ++i) vector2[i] += vector1[i] * scalar;
}

template <typename T> inline void addVector(T& vector1, Real scalar, T& vector2) {
    addVector(vector1.data(), scalar, vector2.data(), vector2.size());
}

template <typename T> inline void addVector(T& vector1, T& vector2) {
    addVector(vector1.data(), 1.0, vector2.data(), vector2.size());
}

template <typename T> inline void addVector(const Feature* vector1, Real scalar, T* vector2, const size_t size) {
    Feature* f = (Feature*)vector1;
    while (f->index != -1 && f->index < size) {
        vector2[f->index] += f->value * scalar;
        ++f;
    }
}

template <typename T> inline void addVector(const Feature* vector1, Real scalar, UnorderedMap<int, T>& vector2) {
    Feature* f = (Feature*)vector1;
    while (f->index != -1) {
        vector2[f->index] += f->value * scalar;
        ++f;
    }
}

template <typename T> inline void addVector(const Feature* vector1, Real scalar, T& vector2) {
    addVector(vector1, scalar, vector2.data(), vector2.size());
}

template <typename T> inline void addVector(const Feature* vector1, T& vector2) {
    addVector(vector1, 1.0, vector2.data(), vector2.size());
}


// Multiply vector by scalar
template <typename T> inline void mulVector(T* vector, Real scalar, const size_t size) {
    for (int f = 0; f < size; ++f) vector[f] *= scalar;
}

template <typename T> inline void mulVector(Feature* vector, Real scalar) {
    for (Feature* f = vector; f->index != -1; ++f) f->value *= scalar;
}

template <typename T> inline void mulVector(T& vector, Real scalar) {
    mulVector(vector.data(), scalar, vector.size());
}

// Divide vector by scalar
inline void divVector(Feature* vector, Real scalar, const size_t size) {
    for (Feature* f = vector; f->index != -1; ++f) f->value /= scalar;
}

inline void divVector(Feature* vector, Real scalar) {
    for (Feature* f = vector; f->index != -1; ++f) f->value /= scalar;
}

template <typename T> inline void divVector(T* vector, Real scalar, const size_t size) {
    for (int f = 0; f < size; ++f) vector[f] /= scalar;
}

template <typename T> inline void divVector(T& vector, Real scalar) {
    divVector(vector.data(), scalar, vector.size());
}

// Unit norm sparse data
template <typename I>
typename std::enable_if<std::is_same<typename std::iterator_traits<I>::value_type, IRVPair>::value, void>::type
unitNorm(I begin, I end) {
    Real norm = 0;
    for (auto i = begin; i != end; ++i) norm += (*i).value * (*i).value;
    if (norm == 0) return;
    norm = std::sqrt(norm);
    for (auto i = begin; i != end; ++i) (*i).value /= norm;
}

// Shift index in sparse data
template <typename I>
typename std::enable_if<std::is_same<typename std::iterator_traits<I>::value_type, IRVPair>::value, void>::type
shift(I begin, I end, int shift) {
    for (auto i = begin; i != end; ++i) (*i).index += shift;
}

// Threshold sparse data
template <typename I>
typename std::enable_if<std::is_same<typename std::iterator_traits<I>::value_type, IRVPair>::value, int>::type
thresholdAbs(I begin, I end, Real threshold) {
    auto n0I = begin;
    int n0 = 0;
    for (auto i = begin; i != end; ++i) {
        if (std::fabs((*i).value) <= threshold) {
            if (n0I != i) {
                (*n0I).index = (*i).index;
                (*n0I).value = (*i).value;
            }
            ++n0I;
            ++n0;
        }
    }
    return n0;
}


class AbstractVector;
class SparseVector;
class MapVector;
class Vector;


// Abstract vector type
class AbstractVector {
public:
    // Constructors
    AbstractVector(): s(0), n0(0) { }
    virtual ~AbstractVector() { }

    // Required to override
    virtual void initD() = 0;
    virtual void insertD(int i, Real v) = 0;
    virtual void checkD() {};

    virtual AbstractVector* copy() = 0;
    virtual void resize(size_t newS) {
        s = newS;
    }
    virtual void reserve(size_t maxN0) {}

    virtual inline Real at(int index) const = 0;
    virtual inline Real& operator[](int index) = 0;
    virtual inline const Real& operator[](int index) const = 0;
    virtual void forEachV(const std::function<void(Real&)>& func) = 0;
    virtual void forEachV(const std::function<void(Real&)>& func) const = 0;
    virtual void forEachIV(const std::function<void(const int&, Real&)>& func) = 0;
    virtual void forEachIV(const std::function<void(const int&, Real&)>& func) const = 0;

    virtual unsigned long long mem() const = 0;

    // Basic math operations, general, slower implementations using forEach
    virtual Real dot(AbstractVector& vec) const;
    virtual Real dot(SparseVector& vec) const;
    virtual Real dot(Feature* vec) const;

    void mul(Real scalar);
    void div(Real scalar);
    void add(Real scalar);
    void add(AbstractVector& vec, Real scalar = 1.0);
    void sub(Real scalar);
    void sub(AbstractVector& vec, Real scalar = 1.0);
    void zero(AbstractVector& vec);
    void invert();
    void zeros();

    virtual void prune(Real threshold);
    virtual void unitNorm();

    inline size_t size() const { return s; }
    inline size_t nonZero() const { return n0; }
    size_t sparseMem() const { return n0 * (sizeof(int) + sizeof(Real)); }
    size_t denseMem() const { return s * sizeof(Real); }

    virtual void save(std::ofstream& out);
    virtual void load(std::ifstream& in);
    static void skipLoad(std::ifstream& in);

    virtual RepresentationType type() const = 0;

    friend std::ostream& operator<<(std::ostream& os, const AbstractVector& vec) {
        os << "{ ";
        vec.forEachIV([&](const int& i, Real& v) {
            os << "(" << i << ", " << v << ") ";
        });
        os << "}";
        return os;
    }

protected:
    size_t s;       // size
    size_t n0;      // non-zero elements
};


class SparseVector: public AbstractVector {
    using AbstractVector::s;
    using AbstractVector::n0;

public:
    SparseVector(): SparseVector(0, 0) { };
    SparseVector(size_t s, size_t maxN0) {
        this->s = s;
        this->maxN0 = maxN0;
        n0 = 0;
        d = new IRVPair[maxN0 + 1];
        sorted = true;
    }
    explicit SparseVector(const AbstractVector& vec) {
        maxN0 = vec.nonZero() + 1;
        d = new IRVPair[maxN0 + 1];
        n0 = 0;
        vec.forEachIV([&](const int& i, Real& v) { insertD(i, v); });
        sort();
    }

    SparseVector(const SparseVector& vec) noexcept {
        if (&vec != this) {
            s = vec.s;
            n0 = vec.n0;
            maxN0 = vec.maxN0;
            sorted = vec.sorted;
            d = new IRVPair[n0 + 1];
            std::copy(vec.begin(), vec.end(), d);
            d[n0].index = -1;
        }
    }

    SparseVector(SparseVector&& vec) noexcept {
        s = vec.s;
        n0 = vec.n0;
        maxN0 = vec.maxN0;
        sorted = vec.sorted;
        d = vec.d;
        vec.d = nullptr;
    }

    explicit SparseVector(const std::vector<IRVPair>& vec, bool sorted = true) {
        s = 0;
        this->sorted = true;
        n0 = vec.size();
        maxN0 = n0;
        d = new IRVPair[n0 + 1];
        d[n0].index = -1;
        if(n0) {
            std::copy(vec.begin(), vec.end(), d);
            this->sorted = sorted;
            sort();
            s = d[n0 - 1].index + 1;
        }
    }

    ~SparseVector() override{
        delete[] d;
    }

    void initD() override {
        delete[] d;
        d = nullptr;
        maxN0 = 0;
        n0 = 0;
        sorted = true;
    }

    void insertD(int i, Real v) override {
        //std::cout << n0 << " " << maxN0 << " " << i << " " << v << "\n";
        if(i >= s) s = i + 1;
        if(v != 0) {
            if(n0 >= maxN0) reserve(2 * maxN0);
            if(v < d[n0].value) sorted = false;
            d[n0++] = {i, v};
            d[n0].index = -1;
        }
    }

    AbstractVector* copy() override {
        auto newVec = new SparseVector(*static_cast<AbstractVector*>(this));
        return static_cast<AbstractVector*>(newVec);
    }

    void reserve(size_t maxN0) override {
        auto newD = new IRVPair[maxN0 + 1]();
        this->maxN0 = maxN0;
        if(d != nullptr){
            std::copy(d, d + std::min(this->n0, maxN0), newD);
            delete[] d;
        }
        d = newD;
        n0 = std::min(this->n0, maxN0);
        d[n0].index = -1;
    }

    Real dot(SparseVector& vec) const override {
        if(sorted && vec.sorted) {
            Real val = 0;

            // Binary search
            auto x = d;
            auto y = vec.d;
            auto xEnd = x + n0;
            auto yEnd = y + vec.n0;
            while(x->index != -1 && y->index != -1){
                if(x->index == y->index){
                    val += x->value * y->value;
                    ++x;
                    ++y;
                }
                else if (x->index < y->index){
                    x = std::lower_bound(x, xEnd, IRVPair(y->index, 0), IRVPairIndexComp());
                }
                else {
                    y = std::lower_bound(y, yEnd, IRVPair(x->index, 0), IRVPairIndexComp());
                }
            }

            // Marching pointers
            /*
            auto x = d;
            for (auto y = vec.data(); y->index != -1; ++y) {
                while (x->index != -1 && x->index < y->index) ++x;
                if (x->index == -1) break;
                else if(x->index == y->index) val += y->value * x->value;
            }
            */

            return val;
        } else return AbstractVector::dot(vec);
    }

    inline Real at(int index) const override {
        auto p = find(index);
        if(p->index == index) return p->value;
        else return 0;
    }

    inline Real& operator[](int index) override {
        return const_cast<Real&>((*const_cast<const SparseVector*>(this))[index]);
    }

    inline const Real& operator[](int index) const override {
        auto p = find(index);
        if(p->index == index) return p->value;
        else return d[n0].value;
    }

    inline const IRVPair* find(int index) const {
        IRVPair* p = d;
        if(sorted) // Binary search
            p = std::lower_bound(d, d + n0, IRVPair(index, 0), IRVPairIndexComp());
        else // Linear search
            while (p->index != -1 && p->index != index) ++p;
        return p;
    }

    void prune(Real threshold) override {
        n0 = thresholdAbs(d, d + n0, threshold);
        d[n0].index = -1;
    }

    void forEachV(const std::function<void(Real&)>& func) override {
        for(auto p = d; p->index != -1; ++p) func(p->value);
    }

    void forEachV(const std::function<void(Real&)>& func) const override {
        for(auto p = d; p->index != -1; ++p) func(p->value);
    }

    void forEachIV(const std::function<void(const int&, Real&)>& func) override {
        for(auto p = d; p->index != -1; ++p) func(p->index, p->value);
    }

    void forEachIV(const std::function<void(const int&, Real&)>& func) const override {
        for(auto p = d; p->index != -1; ++p) func(p->index, p->value);
    }

    unsigned long long mem() const override { return estimateMem(s, n0); };
    static unsigned long long estimateMem(size_t s, size_t n0){
        return sizeof(SparseVector) + n0 * (sizeof(int) + sizeof(Real));
    }

    void load(std::ifstream& in) override {
        AbstractVector::load(in);
        sort();
    }

    bool isSorted() {
        return sorted;
    }

    void sort() {
        if(!sorted){
            std::sort(d, d + n0, IRVPairIndexComp());
            sorted = true;
        }
    }

    RepresentationType type() const override {
        return sparse;
    }

    IRVPair* data(){ return d; }
    IRVPair* begin(){ return d; }
    IRVPair* end(){ return d + n0; }

    const IRVPair* begin() const { return d; }
    const IRVPair* end() const { return d + n0; }

protected:
    size_t maxN0;
    size_t sorted{};
    IRVPair* d; // data
};


class MapVector: public AbstractVector {
    using AbstractVector::s;
    using AbstractVector::n0;

public:
    MapVector(): MapVector(0, 0) { };
    MapVector(size_t s, size_t maxN0): AbstractVector() {
        this->s = s;
        n0 = 0;
        d = new UnorderedMap<int, Real>();
        d->reserve(maxN0);
    }
    explicit MapVector(const AbstractVector& vec) {
        s = vec.size();
        d = new UnorderedMap<int, Real>();
        d->reserve(vec.nonZero());
        vec.forEachIV([&](const int& i, Real& v) { insertD(i, v); });
    }
    ~MapVector() override{
        delete d;
    }

    void initD() override {
        delete d;
        d = new UnorderedMap<int, Real>();
    }

    void insertD(int i, Real v) override {
        if(i >= s) s = i + 1;
        if(v != 0) {
            (*d)[i] = v;
            n0 = d->size();
        }
    }

    void checkD() override {
        n0 = 0;
        forEachIV([&](const int& i, Real& v) {
            if(i >= s) s = i + 1;
            if(v != 0) ++n0;
        });
    };

    Real dot(SparseVector& vec) const override;
    Real dot(Feature* vec) const override;

    AbstractVector* copy() override {
        auto newVec = new MapVector(*static_cast<AbstractVector*>(this));
        return static_cast<AbstractVector*>(newVec);
    }

    void reserve(size_t maxN0) override {
        d->reserve(maxN0);
    }

    inline Real at(int index) const override {
        auto v = d->find(index);
        if (v != d->end()) return v->second;
        else return 0;
    }
    inline const Real& operator[](int index) const override { return (*d)[index]; }
    inline Real& operator[](int index) override { return (*d)[index]; }

    void forEachV(const std::function<void(Real&)>& func) override {
        for (auto& c : *d) func(c.second);
    };

    void forEachV(const std::function<void(Real&)>& func) const override {
        for (auto& c : *d) func(c.second);
    };

    void forEachIV(const std::function<void(const int&, Real&)>& func) override {
        for (auto& c : *d) func(c.first, c.second);
    };

    void forEachIV(const std::function<void(const int&, Real&)>& func) const override {
        for (auto& c : *d) func(c.first, c.second);
    };

    unsigned long long mem() const override {
        unsigned long long mem = sizeof(MapVector);
        if(d != nullptr) mem += d->mask() * (2 * sizeof(int) + sizeof(Real));
        return mem;
    };

    static unsigned long long estimateMem(size_t s, size_t n0){
        unsigned long long mem = sizeof(MapVector);
        size_t mapSize = sizeof(uint64_t);
        while(mapSize < n0) mapSize *= 2;
        mem += mapSize * (2 * sizeof(int) + sizeof(Real));
        return mem;
    }

    RepresentationType type() const override {
        return map;
    }

protected:
    UnorderedMap<int, Real>* d; // data
};


// Simple dense vector
class Vector: public AbstractVector {
    using AbstractVector::s;
    using AbstractVector::n0;

public:
    Vector(): Vector(0) {};
    explicit Vector(size_t s): AbstractVector() {
        this->s = s;
        d = new Real[s]();
    }
    explicit Vector(const AbstractVector& vec): AbstractVector(vec) {
        s = vec.size();
        n0 = 0;
        d = new Real[vec.size()];
        vec.forEachIV([&](const int& i, Real& v) { d[i] = v; });
    }
    ~Vector() override{
        delete d;
    }

    void initD() override {
        if(d != nullptr) delete[] d;
        d = new Real[s]();
        n0 = 0;
    }

    void checkD() override {
        n0 = s;
        for(int i = 0; i < s; ++i){
            if(d[i] == 0) --n0;
        }
    }

    Real dot(Vector& vec) const;
    Real dot(SparseVector& vec) const override;
    Real dot(Feature* vec) const override;

    void insertD(int i, Real v) override {
        if(d[i] == 0 && v != 0) ++n0;
        d[i] = v;
    }

    AbstractVector* copy() override {
        auto newVec = new Vector(*static_cast<AbstractVector*>(this));
        return static_cast<AbstractVector*>(newVec);
    }

    void resize(size_t newS) override {
        auto newD = new Real[newS]();
        if(d != nullptr){
            std::copy(d, d + std::min(s, newS), newD);
            delete[] d;
        }
        s = newS;
        d = newD;
    }

    // Access row also by [] operator
    inline Real at(int index) const override {
        if(index < s) return d[index];
        else return 0;
    }

    inline Real& operator[](int index) override { return d[index]; }
    inline const Real& operator[](int index) const override { return d[index]; }

    void forEachV(const std::function<void(Real&)>& func) override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(d[i]);
    }

    void forEachV(const std::function<void(Real&)>& func) const override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(d[i]);
    }

    void forEachIV(const std::function<void(const int&, Real&)>& func) override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(i, d[i]);
    }

    void forEachIV(const std::function<void(const int&, Real&)>& func) const override {
        for(int i = 0; i < s; ++i) if(d[i] != 0) func(i, d[i]);
    }

    unsigned long long mem() const override { return estimateMem(s, n0); };
    static unsigned long long estimateMem(size_t s, size_t n0){
        return sizeof(Vector) + s * sizeof(Real);
    }

    RepresentationType type() const override {
        return dense;
    }

    Real* data(){ return d; };

    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[ ";
        for (int i = 0; i < v.s; ++i) {
            if (i != 0) os << ", ";
            os << v.d[i];
        }
        os << " ]";
        return os;
    }

protected:
    Real* d; // data
};
