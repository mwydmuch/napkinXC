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

Real Vector::dot(SparseVector& vec) const {
    Real val = 0;
    for(auto &f : vec) val += f.value * d[f.index];
    return val;
}

Real Vector::dot(Feature* vec) const {
    Real val = 0;
    for(auto f = vec; f->index != -1; ++f) val += f->value * d[f->index];
    return val;
}

