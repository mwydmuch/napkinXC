/*
 Copyright (c) 2020 by Marek Wydmuch

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
