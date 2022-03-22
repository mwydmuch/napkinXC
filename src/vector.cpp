#include "vector.h"

Real AbstractVector::dot(AbstractVector& vec) const {
    Real val = 0;
    vec.forEachIV([&](const int& i, Real& v) { val += v * at(i); });
    return val;
}

Real AbstractVector::dot(SparseVector& vec) const {
    Real val = 0;
    for(auto &f : vec) val += f.value * at(f.index);
    return val;
}

Real AbstractVector::dot(Feature* vec) const {
    Real val = 0;
    for(auto f = vec; f->index != -1; ++f) val += f->value * at(f->index);
    return val;
}

void AbstractVector::mul(Real scalar){
    forEachV([&](Real& v) { v *= scalar; });
}

void AbstractVector::div(Real scalar){
    mul(Real(1.0) / scalar);
}

void AbstractVector::add(Real scalar){
    forEachV([&](Real& v) { v += scalar; });
}

void AbstractVector::add(AbstractVector& vec, Real scalar){
    vec.forEachIV([&](const int& i, Real& v) { (*this)[i] += scalar * v; });
}

void AbstractVector::sub(Real scalar){
    add(scalar * Real(-1.0));
}

void AbstractVector::sub(AbstractVector& vec, Real scalar){
    add(vec, scalar * Real(-1.0));
}

void AbstractVector::zero(AbstractVector& vec){
    vec.forEachIV([&](const int& i, Real& v) { (*this)[i] = 0; });
}

void AbstractVector::invert() {
    forEachV([&](Real& v) { v *= -1; });
}

void AbstractVector::zeros(){
    forEachV([&](Real& v) { v = 0; });
}

void AbstractVector::prune(Real threshold){
    forEachIV([&](const int& i, Real& w) {
        if (std::fabs(w) <= threshold) w = 0;
    });
    checkD();
}

void AbstractVector::unitNorm(){
    Real norm = 0;
    forEachV([&](Real& v) { norm += v * v; });
    if (norm == 0) return;
    norm = std::sqrt(norm);
    div(norm);
}

void AbstractVector::save(std::ofstream& out) {
    checkD();
    saveVar(out, s);
    saveVar(out, n0);
    bool sparse = sparseMem() < denseMem() || s == 0; // Select more optimal coding
    saveVar(out, sparse);

    if(sparse) forEachIV([&](const int& i, Real& v) {
        if(v != 0) {
            saveVar(out, i);
            saveVar(out, v);
        }
    });
    else {
        for (int i = 0; i < s; ++i) {
            Real v = at(i);
            saveVar(out, v);
        }
    }
}

void AbstractVector::load(std::ifstream& in) {
    // Load header
    loadVar(in, s);
    size_t n0ToLoad;
    loadVar(in, n0ToLoad);
    bool sparse;
    loadVar(in, sparse);

    // Allocate new vec
    initD(); // Re-init data container
    reserve(n0ToLoad);

    // Load and insert data
    int index;
    Real value;
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
}

void AbstractVector::skipLoad(std::ifstream& in){
    size_t s, n0;
    bool sparse;
    loadVar(in, s);
    loadVar(in, n0);
    loadVar(in, sparse);
    if(sparse) in.seekg(n0 * (sizeof(int) + sizeof(Real)), std::ios::cur);
    else in.seekg(s * sizeof(Real), std::ios::cur);
}

Real MapVector::dot(SparseVector& vec) const {
    Real val = 0;
    for(auto &f : vec) val += f.value * at(f.index);
    return val;
}

Real MapVector::dot(Feature* vec) const {
    Real val = 0;
    for(auto f = vec; f->index != -1; ++f) val += f->value * at(f->index);
    return val;
}

Real Vector::dot(Vector& vec) const {
    size_t minS = std::min(s, vec.s);
    Real val = 0;
    for(size_t i = 0; i < minS; ++i) val += d[i] * vec.d[i];
    return val;
}

Real Vector::dot(SparseVector& vec) const {
    Real val = 0; //TODO: Improve this
    if(vec.size() < s) for(auto &f : vec) val += f.value * d[f.index];
    else for(auto &f : vec) if(f.index < s) val += f.value * d[f.index];
    return val;
}

Real Vector::dot(Feature* vec) const {
    Real val = 0;
    for(auto f = vec; f->index != -1; ++f) val += f->value * d[f->index];
    return val;
}
