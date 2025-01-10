/*
 Copyright (c) 2019-2020 by Marek Wydmuch

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

#include <string>

#include "args.h"
#include "basic_types.h"
#include "vector.h"
#include "matrix.h"

// Libsvm, XMLCRepo and numeric VW data reader
class DataReader {
public:
    DataReader(Args& args);
    virtual ~DataReader() { if(in.is_open()) in.close(); }; 
    bool readData(SRMatrix& labels, SRMatrix& features, Args& args, int rows = -1);
    static void readLine(std::string& line, std::vector<IRVPair>& lLabels, std::vector<IRVPair>& lFeatures);

    static void prepareFeaturesVector(std::vector<IRVPair> &lFeatures, Real bias = 1.0);
    static void processFeaturesVector(std::vector<IRVPair> &lFeatures, bool norm = true, size_t hashSize = 0, Real featuresThreshold = 0);
    static void processLabelsVector(std::vector<IRVPair> &lLabels);

private:
    std::ifstream in;

    int linesRead; // Number of lines read from the file
    int rowsRead; // Number of rows read from the file
    int hLabels;
    int hFeatures;
    int hRows;
    std::string line;
};
