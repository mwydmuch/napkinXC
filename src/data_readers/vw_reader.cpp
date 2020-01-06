/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "vw_reader.h"


VowpalWabbitReader::VowpalWabbitReader() {
    supportHeader = false; // VowpalWabbit format does not have a header
}

VowpalWabbitReader::~VowpalWabbitReader() {}

// Reads line in VowpalWabbit format label,label,... | feature(:value) feature(:value) ...
// Labels and features can be alphanumeric strings
void VowpalWabbitReader::readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures) {
    auto tokens = split(line, '|');
    auto labels = split(tokens[0], ',');
    auto features = split(tokens[1], ' ');

    for (const auto& l : labels) {
        int il = labelsMap.size();
        auto fl = labelsMap.find(l);
        if (fl != labelsMap.end())
            il = fl->second;
        else
            labelsMap.insert({l, il});
        lLabels.push_back(il);
    }

    UnorderedMap<int, double> tmpLFeatures;
    for (auto& f : features) {
        std::string sIndex = f;
        double value = 1.0;

        size_t pos = f.find_first_of(':');
        if (pos != std::string::npos) {
            sIndex = f.substr(0, pos);
            value = std::stof(f.substr(pos + 1, f.length() - pos));
        }

        int index = featuresMap.size() + 2; // Feature (LibLinear ignore feature 0 and feature 1 is reserved for bias)
        auto ff = featuresMap.find(f);
        if (ff != featuresMap.end())
            index = ff->second;
        else
            featuresMap.insert({sIndex, index});
        tmpLFeatures[index] += value;
    }

    for (auto& f : tmpLFeatures) lFeatures.push_back({f.first, f.second});
}

void VowpalWabbitReader::save(std::ostream& out) {
    DataReader::save(out);

    // Save maps
    size_t size = labelsMap.size();
    saveVar(out, size);
    for (auto& l : labelsMap) {
        saveVar(out, l.first);
        saveVar(out, l.second);
    }

    size = featuresMap.size();
    saveVar(out, size);
    for (auto& l : featuresMap) {
        saveVar(out, l.first);
        saveVar(out, l.second);
    }
}

void VowpalWabbitReader::load(std::istream& in) {
    DataReader::load(in);

    // Load maps
    size_t size;
    std::string key;
    int value;

    loadVar(in, size);
    for (int i = 0; i < size; ++i) {
        loadVar(in, key);
        loadVar(in, value);
        labelsMap.insert({key, value});
    }

    loadVar(in, size);
    for (int i = 0; i < size; ++i) {
        loadVar(in, key);
        loadVar(in, value);
        featuresMap.insert({key, value});
    }
}
