/**
 * Copyright (c) 2020 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>

#include "data_reader.h"

class VowpalWabbitReader : public DataReader {
public:
    VowpalWabbitReader();
    ~VowpalWabbitReader() override;

    void readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures) override;

    void save(std::ostream& out) override;
    void load(std::istream& in) override;

private:
    UnorderedMap<std::string, int> labelsMap;
    UnorderedMap<std::string, int> featuresMap;
};
