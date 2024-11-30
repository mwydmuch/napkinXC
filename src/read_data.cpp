/*
 Copyright (c) 2019-2021 by Marek Wydmuch

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

#include <algorithm>

#include "read_data.h"
#include "log.h"
#include "misc.h"


DataReader::DataReader(Args& args) {
    if (args.input.empty())
        throw std::invalid_argument("Empty input path");

    in = std::ifstream(args.input);
    if (!in.is_open())
        throw std::invalid_argument("Cannot open input file: " + args.input);

    hLabels = 0, hFeatures = 0, hRows = 0;

    getline(in, line);
    rowsRead = linesRead = 1;

    auto hTokens = split(line, ' ');
    if(hTokens.size() == 2 || hTokens.size() == 3) {
        hRows = std::stoi(hTokens[0]);
        hFeatures = std::stoi(hTokens[1]);
        getline(in, line);
        ++linesRead;
        if(hTokens.size() == 3) {
            hLabels = std::stoi(hTokens[2]);
            Log(CERR, 2) << "Header detected: rows: " << hRows << ", features: " << hFeatures << ", labels: " << hLabels << "\n";
        } else Log(CERR, 2) << "Header detected: rows: " << hRows << ", features: " << hFeatures << "\n";
    }

    if(args.startRow > args.endRow) 
        throw std::invalid_argument("Start row " + args.startRow + " is bigger then end row " + args.endRow);

    if(args.startRow > 0) {
        while(rowsRead <= args.startRow){
            if(!getline(in, line))
                throw std::invalid_argument("File ended before reaching start row " + args.startRow + ", only " + rowsRead + " rows found");
            ++linesRead;
            ++rowsRead;
        }
    }
}

// Reads train/test data to sparse matrix
bool DataReader::readData(SRMatrix& labels, SRMatrix& features, Args& args, int rows) {
    if (args.hash) hFeatures = args.hash;

    // Read data points
    std::vector<IRVPair> lLabels;
    std::vector<IRVPair> lFeatures;
    int i = 0;
    bool lineRead = true;
    if (!hRows) Log(CERR) << "  ?%\r";
    do {
        if (hRows) printProgress(i, hRows); // If the number of rows is know, print progress

        lLabels.clear();
        lFeatures.clear();

        if(args.processData) prepareFeaturesVector(lFeatures, args.bias);

        try {
            readLine(line, lLabels, lFeatures);
        } catch (const std::exception& e) {
            Log(CERR, 2) << "Failed to read line " << lineRead << ", skipping!\n";
            continue;
        }

        if(args.processData) processFeaturesVector(lFeatures, args.norm, args.hash, args.featuresThreshold);

        labels.appendRow(lLabels);
        features.appendRow(lFeatures);

        ++rowsRead;
        ++linesRead;
        ++i;

        lineRead = getline(in, line) ? true : false;
        if(args.endRow > 0 && rowsRead >= args.endRow) lineRead = false;
        
    } while (lineRead && (rows >= 0 && i < rows));

    // Checks
    assert(labels.rows() == features.rows());
    if (hRows && hRows != features.rows() && rows < 0)
        Log(CERR, 2) << "Warning: Number of lines does not match number in the file header!\n";
    if (hLabels && hFeatures < features.cols() - 2)
        Log(CERR, 2) << "Warning: Number of features is bigger then number in the file header!\n";
    if (hFeatures && hLabels < labels.cols())
        Log(CERR, 2) << "Warning: Number of labels is bigger then number in the file header!\n";

    // Print data
    /*
    for (int r = 0; r < features.rows(); ++r){
        for(int c = 0; c < features.size(r); ++c)
            Log(CERR) << features.row(r)[c].index << ":" << features.row(r)[c].value << " ";
        Log(CERR) << "\n";
    }
    */

    // Print info about loaded data
    Log(CERR) << "Loaded: rows: " << labels.rows() << ", features: " << features.cols() - 2
              << ", labels: " << labels.cols() << "\n  Data size: " << formatMem(labels.mem() + features.mem()) << "\n";

    return lineRead; 
}

// Reads line in LibSvm format label,label,... feature(:value) feature(:value) ...
void DataReader::readLine(std::string& line, std::vector<IRVPair>& lLabels, std::vector<IRVPair>& lFeatures) {
    // Trim leading spaces
    size_t nextPos, pos = line.find_first_not_of(' ');

    while (nextPos = line.find_first_of(",: ", pos)) {
        // Label
        if ((pos == 0 || line[pos - 1] == ',') && (line[nextPos] == ',' || line[nextPos] == ' ' || nextPos == std::string::npos))
            lLabels.emplace_back(std::strtol(&line[pos], NULL, 10), 1.0);

        // Feature index
        else if ((pos == 0 || line[pos - 1] == ' ') && line[nextPos] == ':')
            lFeatures.emplace_back(std::strtol(&line[pos], NULL, 10), 1.0);

        // Feature value
        else if (line[pos - 1] == ':' && (nextPos == std::string::npos || line[nextPos] == ' '))
            lFeatures.back().value = strtor(&line[pos], NULL);

        if (nextPos == std::string::npos) break;
        pos = nextPos + 1;
    }
}

void DataReader::prepareFeaturesVector(std::vector<IRVPair> &lFeatures, Real bias) {
    // Add bias feature (bias feature has index 1)
    lFeatures.emplace_back(1, bias);
}

void DataReader::processFeaturesVector(std::vector<IRVPair> &lFeatures, bool norm, size_t hashSize, Real featuresThreshold) {
    //Shift index by 2 because LibLinear ignore feature 0 and feature 1 is reserved for bias
    //assert(!lFeatures.empty());
    shift(lFeatures.begin() + 1, lFeatures.end(), 2);

    // Hash features
    if (hashSize) {
        UnorderedMap<int, double> lHashed;
        for (int j = 1; j < lFeatures.size(); ++j)
            lHashed[hash(lFeatures[j].index) % hashSize] += lFeatures[j].value;

        lFeatures.erase(lFeatures.begin() + 1, lFeatures.end()); // Keep bias feature
        for (const auto& f : lHashed) lFeatures.emplace_back(f.first + 2, f.second);
    }

    // Norm row
    if (norm) unitNorm(lFeatures.begin() + 1, lFeatures.end()); // Skip bias feature

    // Apply features threshold
    if (featuresThreshold > 0) thresholdAbs(lFeatures.begin() + 1, lFeatures.end(), featuresThreshold);

    // Check if it requires sorting
    if (!std::is_sorted(lFeatures.begin(), lFeatures.end(), IRVPairIndexComp())) sort(lFeatures.begin(), lFeatures.end(), IRVPairIndexComp());
}


