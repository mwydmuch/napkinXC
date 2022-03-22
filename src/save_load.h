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
#include <iostream>

// Simple save/load utils
class FileHelper {
public:
    void saveToFile(std::string outfile);
    virtual void save(std::ofstream& out) = 0;
    void loadFromFile(std::string infile);
    virtual void load(std::ifstream& in) = 0;
};

template <typename T> inline void saveVar(std::ofstream& out, T& var) { out.write((char*)&var, sizeof(T)); }

template <typename T> inline void loadVar(std::ifstream& in, T& var) { in.read((char*)&var, sizeof(T)); }

inline void saveVar(std::ofstream& out, std::string& var) {
    size_t size = var.size();
    out.write((char*)&size, sizeof(size));
    out.write((char*)&var[0], size);
}

inline void loadVar(std::ifstream& in, std::string& var) {
    size_t size;
    in.read((char*)&size, sizeof(size));
    var.resize(size);
    in.read((char*)&var[0], size);
}
