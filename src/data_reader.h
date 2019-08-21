/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>

#include "args.h"
#include "types.h"
#include "utils.h"

class DataReader: public FileHelper{
public:
    DataReader();
    virtual ~DataReader();

    void readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    virtual void readHeader(std::string& line) = 0;
    virtual void readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures) = 0;

    void save(std::ostream& out) override;
    void load(std::istream& in) override;

protected:
    int hLabels;
    int hFeatures;
    int hRows;
};

std::shared_ptr<DataReader> dataReaderFactory(Args &args);
