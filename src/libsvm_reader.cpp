/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <string>

#include "libsvm_reader.h"
#include "types.h"
#include "utils.h"

LibSvmReader::LibSvmReader(){}

LibSvmReader::~LibSvmReader(){}

// Read header in LibSvm Format: #rows #features #labels
void LibSvmReader::readHeader(std::string &line){
    int hRows = -1;
    size_t nextPos, pos = 0;

    nextPos = line.find_first_of(" ", pos);
    hRows = std::stoi(line.substr(pos, nextPos - pos));
    pos = nextPos + 1;

    nextPos = line.find_first_of(" ", pos);
    if(!hFeatures) hFeatures = std::stoi(line.substr(pos, nextPos - pos));
    pos = nextPos + 1;

    nextPos = line.find_first_of(" ", pos);
    if(!hLabels) hLabels = std::stoi(line.substr(pos, nextPos - pos));

    std::cerr << "  Header: rows: " << hRows << ", features: " << hFeatures << ", labels: " << hLabels << std::endl;
}


// Reads line in LibSvm format label,label,... feature(:value) feature(:value) ...
void LibSvmReader::readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures){
    size_t nextPos, pos = line[0] == ' ' ? 1 : 0;
    bool requiresSort = false;

    while((nextPos = line.find_first_of(",: ", pos))){
        // Label
        if((pos == 0 || line[pos - 1] == ',') && (line[nextPos] == ',' || line[nextPos] == ' '))
            lLabels.push_back(std::stoi(line.substr(pos, nextPos - pos)));

            // Feature (LibLinear ignore feature 0)
        else if(line[pos - 1] == ' ' && line[nextPos] == ':')
            lFeatures.push_back({std::stoi(line.substr(pos, nextPos - pos)) + 1, 1.0});

        else if(line[pos - 1] == ':' && (line[nextPos] == ' ' || nextPos == std::string::npos))
            lFeatures.back().value = std::stof(line.substr(pos, nextPos - pos));

        if(nextPos == std::string::npos) break;
        pos = nextPos + 1;
    }
}
