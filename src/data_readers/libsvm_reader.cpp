/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "libsvm_reader.h"
#include "misc.h"


LibSvmReader::LibSvmReader() {
    supportHeader = true;
}

LibSvmReader::~LibSvmReader() {}

// Read header in LibSvm Format: #rows #features #labels
void LibSvmReader::readHeader(std::string& line, int& hLabels, int& hFeatures, int& hRows) {
    auto hTokens = split(line, ' ');
    hRows = std::stoi(hTokens[0]);
    hFeatures = std::stoi(hTokens[1]);
    hLabels = std::stoi(hTokens[2]);
}

// Reads line in LibSvm format label,label,... feature(:value) feature(:value) ...
// TODO: rewrite this using split?
void LibSvmReader::readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures) {
    // Trim leading spaces
    size_t nextPos, pos = line.find_first_not_of(' ');

    while ((nextPos = line.find_first_of(",: ", pos))) {
        // Label
        if ((pos == 0 || line[pos - 1] == ',') && (line[nextPos] == ',' || line[nextPos] == ' ' || nextPos == std::string::npos))
            lLabels.push_back(std::stoi(line.substr(pos, nextPos - pos)));

        // Feature index
        else if (line[pos - 1] == ' ' && line[nextPos] == ':') {
            int index = std::stoi(line.substr(pos, nextPos - pos)) + 2; // Feature (LibLinear ignore feature 0 and feature 1 is reserved for bias)
            lFeatures.push_back({index, 1.0});
        }

        // Feature value
        else if (line[pos - 1] == ':' && (line[nextPos] == ' ' || nextPos == std::string::npos))
            lFeatures.back().value = std::stof(line.substr(pos, nextPos - pos));

        if (nextPos == std::string::npos) break;
        pos = nextPos + 1;
    }
}
