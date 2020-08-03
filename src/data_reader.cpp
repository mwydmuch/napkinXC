/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <unordered_map>

#include "data_reader.h"
#include "libsvm_reader.h"
#include "misc.h"
#include "vw_reader.h"


std::shared_ptr<DataReader> DataReader::factory(Args& args) {
    std::shared_ptr<DataReader> dataReader = nullptr;
    switch (args.dataFormatType) {
    case libsvm: dataReader = std::static_pointer_cast<DataReader>(std::make_shared<LibSvmReader>()); break;
    case vw: dataReader = std::static_pointer_cast<DataReader>(std::make_shared<VowpalWabbitReader>()); break;
    default: throw std::invalid_argument("Unknown data reader type!");
    }

    return dataReader;
}

DataReader::DataReader() { supportHeader = false; }

DataReader::~DataReader() {}

void DataReader::readHeader(std::string& line, int& hLabels, int& hFeatures, int& hRows) {}

// Reads train/test data to sparse matrix
void DataReader::readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    std::cerr << "Loading data from: " << args.input << std::endl;

    std::ifstream in;
    in.open(args.input);
    std::string line;

    // Read header
    int i = 1;
    int hLabels = 0, hFeatures = 0, hRows = 0;
    if (args.header && supportHeader) {
        getline(in, line);
        ++i;
        try {
            readHeader(line, hLabels, hFeatures, hRows);
            std::cerr << "  Header: rows: " << hRows << ", features: " << hFeatures << ", labels: " << hLabels
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Failed to read header from input!\n";
            exit(1);
        }
    }
    if (args.hash) hFeatures = args.hash;

    // Read data points
    std::vector<Label> lLabels;
    std::vector<Feature> lFeatures;
    if (!hRows) std::cerr << "  ?%\r";

    while (getline(in, line)) {
        if (hRows) printProgress(i++, hRows); // If the number of rows is know, print progress

        lLabels.clear();
        lFeatures.clear();

        prepareFeaturesVector(lFeatures, args.bias);

        try {
            readLine(line, lLabels, lFeatures);
        } catch (const std::exception& e) {
            std::cerr << "  Failed to read line " << i << " from input!\n";
            exit(1);
        }

        processFeaturesVector(lFeatures, args.norm, args.hash, args.featuresThreshold);

        labels.appendRow(lLabels);
        features.appendRow(lFeatures);
    }

    in.close();

    // Checks
    assert(labels.rows() == features.rows());
    if (args.header && supportHeader) {
        if (hRows != features.rows())
            std::cerr << "  Warning: Number of lines does not match number in the file header!\n";
        if (hLabels != labels.cols())
            std::cerr << "  Warning: Number of labels does not match number in the file header!\n";
        if (hFeatures != features.cols() - 2)
            std::cerr << "  Warning: Number of features does not match number in the file header!\n";
    }

    // Print data
    /*
    for (int r = 0; r < features.rows(); ++r){
       for(int c = 0; c < features.size(r); ++c)
           std::cerr << features.row(r)[c].index << ":" << features.row(r)[c].value << " ";
       std::cerr << "\n";
    }
    */

    // Print info about loaded data
    std::cerr << "  Loaded: rows: " << labels.rows() << ", features: " << features.cols() - 2
              << ", labels: " << labels.cols() << "\n  Data size: " << formatMem(labels.mem() + features.mem()) << std::endl;
}

void DataReader::processFeaturesVector(std::vector<Feature> &lFeatures, bool norm, int hashSize, double featuresThreshold) {
    // Hash features
    if (hashSize) {
        UnorderedMap<int, double> lHashed;
        for (int j = 1; j < lFeatures.size(); ++j)
            lHashed[hash(lFeatures[j].index) % hashSize] += lFeatures[j].value;

        lFeatures.erase (lFeatures.begin() + 1, lFeatures.end()); // Keep bias feature
        for (const auto& f : lHashed) lFeatures.push_back({f.first + 2, f.second});
    }

    // Norm row
    if (norm) unitNorm(lFeatures);

    // Apply features threshold
    if (featuresThreshold > 0) threshold(lFeatures, featuresThreshold);

    // Check if it requires sorting
    if (!std::is_sorted(lFeatures.begin(), lFeatures.end())) sort(lFeatures.begin(), lFeatures.end());
}

void DataReader::save(std::ostream& out) {}

void DataReader::load(std::istream& in) {}
