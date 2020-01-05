/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>

#include "data_reader.h"

class LibSvmReader : public DataReader {
public:
    LibSvmReader();
    ~LibSvmReader() override;

    void readHeader(std::string& line, int& hLabels, int& hFeatures, int& hRows) override;
    void readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures) override;
};
