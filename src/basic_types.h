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

#include "robin_hood.h"
#include "save_load.h"

// Basic types
typedef float Real;
#define strtor std::strtof

template<typename T>
struct IVPair{
    int index;
    T value;

    IVPair(): index(0), value(0) {}
    IVPair(int index, float value): index(index), value(value) {}
    
    bool operator<(const IVPair<T>& r) const { return value < r.value; };

    friend std::ostream& operator<<(std::ostream& os, const IVPair<T>& fn) {
        os << fn.index << ":" << fn.value;
        return os;
    }
};


typedef IVPair<Real> IRVPair;
typedef IVPair<int> IIVPair;

typedef IRVPair Feature;

// TODO: Replace prediction with IRVPair
//typedef IRVPair Prediction;

struct Prediction{
    int label;
    Real value; // labels's value/probability/loss

    Prediction(): label(0), value(0) {}
    Prediction(int label, double value): label(label), value(value) {};

    bool operator<(const Prediction& r) const { return value < r.value; };

    friend std::ostream& operator<<(std::ostream& os, const Prediction& fn) {
        os << fn.label << ":" << fn.value;
        return os;
    }
};

typedef int Label;
#define UnorderedMap robin_hood::unordered_flat_map
#define UnorderedSet robin_hood::unordered_flat_set


// Helpers - comperators
template<typename T>
struct IVPairIndexComp{
    bool operator()(IVPair<T> const& lhs, IVPair<T> const& rhs){
        return lhs.index < rhs.index;
    }
};

template<typename T>
struct IVPairValueComp{
    bool operator()(IVPair<T> const& lhs, IVPair<T> const& rhs){
        return lhs.value < rhs.value;
    }
};

typedef IVPairIndexComp<Real> IRVPairIndexComp;
typedef IVPairValueComp<Real> IRVPairValueComp;
typedef IVPairIndexComp<Real> IIRVPairIndexComp;
typedef IVPairValueComp<Real> IIVPairValueComp;


template <typename T, typename U>
struct pairFirstComp{
    bool operator()(std::pair<T, U> const& lhs, std::pair<T, U> const& rhs){
        return lhs.first < rhs.first;
    }
};


template <typename T, typename U>
struct pairSecondComp{
    bool operator()(std::pair<Real, U> const& lhs, std::pair<T, U> const& rhs){
        return lhs.second < rhs.second;
    }
};


// Helpers - out operators
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


// TopKQueue
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
