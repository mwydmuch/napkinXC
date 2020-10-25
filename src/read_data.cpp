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

#include <algorithm>

#include "read_data.h"
#include "log.h"


// Reads train/test data to sparse matrix
void readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    if (args.input.empty())
        throw std::invalid_argument("Empty input path");

    Log(CERR) << "Loading data from: " << args.input << "\n";

    std::ifstream in;
    in.open(args.input);
    std::string line;

    // Check header
    int i = 1; // Line counter
    int hLabels = 0, hFeatures = 0, hRows = 0;
    getline(in, line);

    auto hTokens = split(line, ' ');
    if(hTokens.size() == 3) {
        hRows = std::stoi(hTokens[0]);
        hFeatures = std::stoi(hTokens[1]);
        hLabels = std::stoi(hTokens[2]);
        getline(in, line);
        ++i;
        Log(CERR) << "  Header: rows: " << hRows << ", features: " << hFeatures << ", labels: " << hLabels << "\n";
    }
    if (args.hash) hFeatures = args.hash;

    // Read data points
    std::vector<Label> lLabels;
    std::vector<Feature> lFeatures;
    if (!hRows) Log(CERR) << "  ?%\r";
    do {
        if (hRows) printProgress(i, hRows); // If the number of rows is know, print progress

        lLabels.clear();
        lFeatures.clear();

        prepareFeaturesVector(lFeatures, args.bias);

        try {
            readLine(line, lLabels, lFeatures);
        } catch (const std::exception& e) {
            Log(CERR) << "  Failed to read line " << i << " from input!\n";
            exit(1);
        }

        processFeaturesVector(lFeatures, args.norm, args.hash, args.featuresThreshold);

        labels.appendRow(lLabels);
        features.appendRow(lFeatures);

        ++i;
    } while (getline(in, line));

    in.close();

    // Checks
    assert(labels.rows() == features.rows());
    if (hRows && hLabels && hFeatures) {
        if (hRows != features.rows())
            Log(CERR) << "  Warning: Number of lines does not match number in the file header!\n";
        if (hLabels < labels.cols())
            Log(CERR) << "  Warning: Number of labels is bigger then number in the file header!\n";
        if (hFeatures < features.cols() - 2)
            Log(CERR) << "  Warning: Number of features is bigger then number in the file header!\n";
    }

    // Print data
    /*
    for (int r = 0; r < features.rows(); ++r){
        for(int c = 0; c < features.size(r); ++c)
            Log(CERR) << features.row(r)[c].index << ":" << features.row(r)[c].value << " ";
        Log(CERR) << "\n";
    }
    */

    // Print info about loaded data
    Log(CERR) << "  Loaded: rows: " << labels.rows() << ", features: " << features.cols() - 2
              << ", labels: " << labels.cols() << "\n  Data size: " << formatMem(labels.mem() + features.mem()) << "\n";
}

// Reads line in LibSvm format label,label,... feature(:value) feature(:value) ...
// TODO: rewrite this using split?
void readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures) {
    // Trim leading spaces
    size_t nextPos, pos = line.find_first_not_of(' ');

    while ((nextPos = line.find_first_of(",: ", pos))) {
        // Label
        if ((pos == 0 || line[pos - 1] == ',') &&
            (line[nextPos] == ',' || line[nextPos] == ' ')) // || nextPos == std::string::npos))
            lLabels.push_back(std::stoi(line.substr(pos, nextPos - pos)));

            // Feature index
        else if (line[pos - 1] == ' ' && line[nextPos] == ':') {
            // Feature (LibLinear ignore feature 0 and feature 1 is reserved for bias)
            int index = std::stoi(line.substr(pos, nextPos - pos)) + 2;
            lFeatures.push_back({index, 1.0});
        }

            // Feature value
        else if (line[pos - 1] == ':' && (line[nextPos] == ' ' || nextPos == std::string::npos))
            lFeatures.back().value = std::stof(line.substr(pos, nextPos - pos));

        if (nextPos == std::string::npos) break;
        pos = nextPos + 1;
    }
}

void prepareFeaturesVector(std::vector<Feature> &lFeatures, double bias) {
    // Add bias feature (bias feature has index 1)
    lFeatures.push_back({1, bias});
}

void processFeaturesVector(std::vector<Feature> &lFeatures, bool norm, int hashSize, double featuresThreshold) {
    // Hash features
    if (hashSize) {
        UnorderedMap<int, double> lHashed;
        for (int j = 1; j < lFeatures.size(); ++j)
            lHashed[hash(lFeatures[j].index) % hashSize] += lFeatures[j].value;

        lFeatures.erase (lFeatures.begin() + 1, lFeatures.end()); // Keep bias feature
        for (const auto& f : lHashed) lFeatures.push_back({f.first + 2, f.second});
    }

    // Norm row
    if (norm) unitNorm(lFeatures.data() + 1, lFeatures.size() - 1);

    // Apply features threshold
    if (featuresThreshold > 0) threshold(lFeatures, featuresThreshold);

    // Check if it requires sorting //TODO: Move this to matrix class
    if (!std::is_sorted(lFeatures.begin(), lFeatures.end())) sort(lFeatures.begin(), lFeatures.end());
}


